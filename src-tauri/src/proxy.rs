use std::{collections::HashMap, sync::Arc};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::{net::TcpListener, sync::Mutex};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    config::{ConfigStore, ProviderKind, RouteEvent},
    router::{decide, ResolvedRoute, RoutePreviewInput},
    protocol::{self, Protocol},
};

#[derive(Clone)]
struct ProxyContext {
    store: Arc<ConfigStore>,
    client: Client,
    sessions: Arc<Mutex<HashMap<String, ResolvedRoute>>>,
    health: crate::resilience::Health,
}

pub struct ProxyHandle {
    paused: Arc<std::sync::atomic::AtomicBool>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    pub health: crate::resilience::Health,
    task: tokio::task::JoinHandle<()>,
}

impl ProxyHandle {
    pub fn running(&self)->bool{!self.task.is_finished()}
    pub fn paused(&self) -> bool { self.paused.load(std::sync::atomic::Ordering::SeqCst) }
    pub fn set_paused(&self, paused: bool) { self.paused.store(paused, std::sync::atomic::Ordering::SeqCst); }
    pub async fn stop(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if tokio::time::timeout(std::time::Duration::from_secs(5), &mut self.task).await.is_err() { self.task.abort(); }
    }
}

pub async fn start(store: Arc<ConfigStore>) -> anyhow::Result<ProxyHandle> {
    start_on(store, "127.0.0.1").await
}

/// Bind the gateway to an explicit interface. The desktop app keeps using [`start`],
/// which stays on loopback; a server deployment passes its own host (usually 0.0.0.0).
pub async fn start_on(store: Arc<ConfigStore>, host: &str) -> anyhow::Result<ProxyHandle> {
    let listener = TcpListener::bind((host, store.read().port)).await?;
    let circuit_health = crate::resilience::Health::default();
    let client = store.read().gateway.client()?;
    let context = ProxyContext {
        health: circuit_health.clone(),
        store,
        client,
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };
    let cors = CorsLayer::new()
        .allow_origin([
            "http://localhost".parse::<HeaderValue>()?,
            "tauri://localhost".parse::<HeaderValue>()?,
        ])
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE, header::ACCEPT]);
    let paused = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(model_catalog))
        .route("/v1beta/models/{*operation}", post(gemini))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(axum::middleware::from_fn_with_state(paused.clone(), pause_requests))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(context);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let server = axum::serve(listener, app).with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        });
        if let Err(error) = server.await {
            eprintln!("AutoJev proxy stopped: {error}");
        }
    });
    Ok(ProxyHandle {
        health: circuit_health, task, paused,
        shutdown: Some(shutdown_tx),
    })
}

fn paused_response() -> Response {
    (StatusCode::NOT_FOUND, [("x-should-retry", "false")], Json(json!({"error":{"type":"not_found_error","code":"gateway_paused","message":"AutoJev 网关已暂停，请在 AutoJev 中恢复服务后重试。Gateway paused; resume AutoJev before sending another request."}}))).into_response()
}
async fn pause_requests(State(paused): State<Arc<std::sync::atomic::AtomicBool>>, request: axum::extract::Request, next: axum::middleware::Next) -> Response {
    if paused.load(std::sync::atomic::Ordering::SeqCst) && request.method() == Method::POST { return paused_response(); }
    next.run(request).await
}

async fn health(State(context): State<ProxyContext>) -> impl IntoResponse {
    let config = context.store.read();
    Json(json!({
        "status": "ok",
        "service": "autojev-local-router",
        "version": env!("CARGO_PKG_VERSION"),
        "models": config.models.iter().filter(|model| model.enabled).count()
    }))
}

fn rejected_request(context: &ProxyContext, headers: &HeaderMap, endpoint: &str,
    error: axum::extract::rejection::JsonRejection) -> Response {
    let capture = crate::traffic::Capture::new(endpoint, &json!({}), headers);
    capture.lock().unwrap().log.error = "Invalid JSON request body or content type".into();
    crate::traffic::response(error.into_response(), capture, context.store.clone())
}

async fn chat_completions(
    State(context): State<ProxyContext>,
    headers: HeaderMap,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body { Ok(Json(body)) => body, Err(error) => return rejected_request(&context, &headers, "chat/completions", error) };
    forward(
        context,
        headers,
        body,
        "/v1/chat/completions",
        "chat/completions",
    )
    .await
}

async fn gemini(
    State(context): State<ProxyContext>,
    axum::extract::Path(operation): axum::extract::Path<String>,
    headers: HeaderMap, body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body { Ok(Json(body)) => body, Err(error) => return rejected_request(&context, &headers, &format!("v1beta/models/{operation}"), error) };
    let mut metadata = json!({});
    metadata["model"] = json!(operation.rsplit_once(':').map(|(m,_)|m).unwrap_or(""));
    metadata["stream"] = json!(operation.ends_with(":streamGenerateContent"));
    let capture = crate::traffic::Capture::new(&format!("v1beta/models/{operation}"), &metadata, &headers);
    let store = context.store.clone();
    let response = gemini_captured(context, operation, headers, body, capture.clone()).await;
    crate::traffic::response(response, capture, store)
}

async fn gemini_captured(context: ProxyContext, operation: String, headers: HeaderMap, body: Value,
    capture: crate::traffic::SharedCapture) -> Response {
    let Some((model, action)) = operation.rsplit_once(':') else {return error_response(StatusCode::BAD_REQUEST,"Invalid Gemini action");};
    if action == "countTokens" {
        return Json(json!({"totalTokens": (body.to_string().chars().count() / 4).max(1)})).into_response();
    }
    if !matches!(action,"generateContent"|"streamGenerateContent") {
        return error_response(StatusCode::NOT_IMPLEMENTED,"Unsupported Gemini action");
    }
    let mapped = match crate::gemini_bridge::request(model, &body) {
        Ok(v)=>v,Err(e)=>return error_response(StatusCode::BAD_REQUEST,&e.to_string()),
    };
    let response = forward_captured(context, headers, mapped, "chat/completions", capture).await;
    if !response.status().is_success(){return response;}
    let bytes = match axum::body::to_bytes(response.into_body(),32*1024*1024).await {
        Ok(v)=>v,Err(_)=>return error_response(StatusCode::BAD_GATEWAY,"Upstream response too large"),
    };
    let converted = serde_json::from_slice(&bytes).map_err(anyhow::Error::from).and_then(|v|crate::gemini_bridge::response(&v));
    match converted {
        Ok(v) if action=="streamGenerateContent" => (
            [(header::CONTENT_TYPE,"text/event-stream"),(header::CACHE_CONTROL,"no-cache")],
            format!("data: {}\n\n",v)
        ).into_response(),
        Ok(v)=>Json(v).into_response(),
        Err(_)=>error_response(StatusCode::BAD_GATEWAY,"Invalid upstream completion"),
    }
}

async fn responses(
    State(context): State<ProxyContext>,
    headers: HeaderMap,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body { Ok(Json(body)) => body, Err(error) => return rejected_request(&context, &headers, "responses", error) };
    forward(context, headers, body, "/v1/responses", "responses").await
}

async fn messages(
    State(context): State<ProxyContext>,
    headers: HeaderMap,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body { Ok(Json(body)) => body, Err(error) => return rejected_request(&context, &headers, "messages", error) };
    forward(context, headers, body, "/v1/messages", "messages").await
}

async fn forward(
    context: ProxyContext,
    headers: HeaderMap,
    body: Value,
    _upstream_path: &str,
    endpoint: &str,
) -> Response {
    let capture = crate::traffic::Capture::new(endpoint, &body, &headers);
    let store = context.store.clone();
    let response = forward_captured(context, headers, body, endpoint, capture.clone()).await;
    crate::traffic::response(response, capture, store)
}

async fn forward_captured(context: ProxyContext, headers: HeaderMap, body: Value, endpoint: &str,
    capture: crate::traffic::SharedCapture) -> Response {
    let config = context.store.read();
    let canonical = crate::router::normalize_requested_model(&config, body["model"].as_str()).ok().flatten().unwrap_or_default();
    let mut binding = canonical.strip_prefix("autojev/").unwrap_or("").to_owned();
    if let Some(value) = headers.get("x-autojev-binding").and_then(|v|v.to_str().ok()) { binding = value.into(); }
    if let Some(agent) = headers.get("x-autojev-agent").and_then(|v|v.to_str().ok()) {
        if let Some(entry) = config.agent_catalogs.get(agent).and_then(|c|c.iter().find(|e| e.id == body["model"].as_str().unwrap_or(""))) {binding = entry.binding.clone();}
    }
    let attempts = config.routes.iter().find(|r| r.id == binding && r.enabled && matches!(r.strategy.as_str(), "round_robin" | "jev"))
        .map_or(1, |r| config.models.iter().filter(|m| r.includes_model(&m.id) && m.enabled && config.providers.iter().any(|p|p.id == m.provider_id && p.enabled)).count().max(1));
    let attempts = attempts.min(config.gateway.max_attempts);
    let mut tried = std::collections::HashSet::new();
    let mut last_response = None;
    for attempt in 0..attempts {
        let before = tried.len();
        let mut lease = None;
        let started = std::time::Instant::now();
        let mut response = forward_attempt(context.clone(), headers.clone(), body.clone(), endpoint, capture.clone(), &mut tried, &mut lease).await;
        if tried.len() == before && last_response.is_some() {return last_response.unwrap();}
        let status = response.status().as_u16();
        if let Some(lease) = lease {
            if response.status().is_success() {response=observe_health(response,lease,config.gateway.clone(),capture.clone());}
            else {lease.complete(status, crate::resilience::retry_after(response.headers()), &config.gateway);}
        }
        { let mut c = capture.lock().unwrap();
          let provider = c.log.provider_name.clone(); let model = c.log.model_id.clone();
          c.log.attempts.push(crate::traffic::Attempt {provider,model,status,duration_ms:started.elapsed().as_millis() as u64});
        }
        let retry = crate::resilience::retryable(status);
        if !retry || attempt + 1 == attempts {return response;}
        let failed_sample = {
            let mut c = capture.lock().unwrap();
            let mut log = c.log.clone();
            log.status = "error".into();
            log.upstream_duration_ms = started.elapsed().as_millis() as u64;
            c.log.performance_model_id.clear();
            log
        };
        if let Err(error) = crate::performance::record_log(&context.store,&failed_sample,false) {
            eprintln!("Could not persist failed attempt performance: {error}");
        }
        last_response = Some(response);
    }
    unreachable!()
}

fn request_compatible(body: &Value, source: Protocol, model: &crate::config::Model, provider: &crate::config::Provider) -> anyhow::Result<()> {
    Protocol::upstream(model,provider).and_then(|target| protocol::convert_request(body,source,target,&model.model_id).map(|_|()))
}

async fn forward_attempt(context: ProxyContext, headers: HeaderMap, body: Value, endpoint: &str,
    capture: crate::traffic::SharedCapture, tried: &mut std::collections::HashSet<String>, lease: &mut Option<crate::resilience::Lease>) -> Response {
    let mut input = inspect_request(&body, endpoint);
    if let Some(binding) = headers.get("x-autojev-binding").and_then(|v| v.to_str().ok()) {
        input.requested_model = Some(format!("autojev/{binding}"));
    }
    let mut config = context.store.read();
    crate::traffic::resolve_requested_model(&mut capture.lock().unwrap().log, &config, input.requested_model.as_deref().unwrap_or(""));
    let session_config = serde_json::to_string(&(&config.routes, &config.models, &config.providers, &config.policy)).unwrap_or_default();
    config.models.retain(|m| !tried.contains(&m.id) && context.health.available(&m.id));
    if config.models.is_empty() {
        let mut response = error_response(StatusCode::SERVICE_UNAVAILABLE, "All candidate models are cooling down or unavailable. Retry shortly.");
        response.headers_mut().insert("retry-after", HeaderValue::from_static("5"));
        return response;
    }
    if let Some(agent) = headers.get("x-autojev-agent").and_then(|v| v.to_str().ok()) {
        let requested = body["model"].as_str().unwrap_or("");
        match config.agent_catalogs.get(agent).and_then(|items| items.iter().find(|entry| entry.id == requested)) {
            Some(entry) => input.requested_model = Some(format!("autojev/{}", entry.binding)),
            None => return error_response(StatusCode::UNPROCESSABLE_ENTITY, "Model is not in this agent's selected model list"),
        }
    }
    input.requested_model = match crate::router::normalize_requested_model(&context.store.read(), input.requested_model.as_deref()) {
        Ok(model) => model,
        Err(error) => return error_response(StatusCode::UNPROCESSABLE_ENTITY, &error.to_string()),
    };
    capture.lock().unwrap().routing_rule(&config, input.requested_model.as_deref().unwrap_or(""));
    let requested=input.requested_model.as_deref().unwrap_or("").strip_prefix("autojev/").unwrap_or("");
    let unavailable = if let Some(id)=requested.strip_prefix("model/") {context.store.read().models.iter().any(|m|m.id==id) && !config.models.iter().any(|m|m.id==id)} else {config.routes.iter().find(|r|r.id==requested && r.enabled).is_some_and(|r|!config.models.iter().any(|m|r.includes_model(&m.id)))};
    if unavailable {let mut response=error_response(StatusCode::SERVICE_UNAVAILABLE,"All candidates for this route are cooling down or already attempted. Retry shortly.");response.headers_mut().insert("retry-after",HeaderValue::from_static("5"));return response;}
    // Validate the actual payload before selecting (or reusing) a model. Native
    // hosted tools cannot be implemented merely by translating the JSON schema.
    let source = match Protocol::parse(endpoint) {
        Ok(protocol) => protocol,
        Err(error) => return error_response(StatusCode::UNPROCESSABLE_ENTITY,&error.to_string()),
    };
    let binding=input.requested_model.as_deref().unwrap_or("").strip_prefix("autojev/");
    let mut conversion_error=None;
    config.models.retain(|model| {
        let in_scope=match binding {
            Some(id) if id.starts_with("model/") => model.id==id.trim_start_matches("model/"),
            Some(id) => config.routes.iter().find(|r|r.id==id).is_none_or(|r|r.includes_model(&model.id)),
            None => true,
        };
        if !in_scope { return true; }
        let Some(provider)=config.providers.iter().find(|p|p.id==model.provider_id) else { return false; };
        match request_compatible(&body,source,model,provider) {
            Ok(()) => true,
            Err(error) => {conversion_error=Some(error.to_string());false},
        }
    });
    if let Some(error)=conversion_error {
        let scoped_available=config.models.iter().any(|model| match binding {
            Some(id) if id.starts_with("model/") => model.id==id.trim_start_matches("model/"),
            Some(id) => config.routes.iter().find(|r|r.id==id).is_none_or(|r|r.includes_model(&model.id)),
            None => true,
        });
        if !scoped_available { return error_response(StatusCode::UNPROCESSABLE_ENTITY,&format!("No candidate supports this request. {error} For Codex through AutoJev, reconnect and restart Codex to apply web_search=disabled, or choose a native Responses model supporting this tool.")); }
    }
    use std::hash::{Hash, Hasher};
    let mut version = std::collections::hash_map::DefaultHasher::new();
    session_config.hash(&mut version);
    let session_id = session_id(&headers, &body).map(|id| format!("{}:{}:{}:{}", id, endpoint,
        input.requested_model.as_deref().unwrap_or(""), version.finish()));


    let resolved = if let Some(id) = session_id.as_ref() {
        let sessions = context.sessions.lock().await;
        sessions
            .get(id)
            .filter(|route| still_eligible(route, &input, &config))
            .cloned()
    } else {
        None
    };

    if crate::cost::is_cost_route(&config, &input) {
        let store = context.store.clone();
        config.cost_history = tauri::async_runtime::spawn_blocking(move || store.recent_cost_logs().unwrap_or_default()).await.unwrap_or_default();
        config.output_limit = body.get("max_completion_tokens").or_else(||body.get("max_output_tokens")).or_else(||body.get("max_tokens")).and_then(Value::as_u64);
        config.cost_incumbent = resolved.as_ref().map(|r|r.model.id.clone());
    }
    let resolved = if crate::cost::is_cost_route(&config, &input) { None } else { resolved };
    let resolved = match resolved {
        Some(mut route) => {
            route.decision.reason = "Kept the model selected for this agent session.".into();
            route
        }
        None => match decide(&config, &input, &context.client, context.store.read_secret("autojev-cloud").as_deref()).await {
            Ok(route) => {
                if let Some(id) = session_id {
                    let mut sessions = context.sessions.lock().await;
                    if sessions.len() >= 4096 { sessions.clear(); }
                    sessions.insert(id, route.clone());
                }
                route
            }
            Err(error) => {
                return error_response(StatusCode::UNPROCESSABLE_ENTITY, &error.to_string())
            }
        },
    };

    *lease = context.health.acquire(&resolved.model.id);
    if lease.is_none() {return error_response(StatusCode::SERVICE_UNAVAILABLE, "Candidate is being probed by another request. Retry shortly.");}
    tried.insert(resolved.model.id.clone());
    { let mut c=capture.lock().unwrap();c.route(&resolved);c.log.performance_context_tokens=input.estimated_context_tokens;c.log.performance_requires_vision=input.requires_vision; }
    let source = match Protocol::parse(endpoint) {
        Ok(protocol) => protocol,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let target = match Protocol::upstream(&resolved.model, &resolved.provider) {
        Ok(protocol) => protocol,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, Json(source.error(&error.to_string()))).into_response(),
    };
    let streaming = body["stream"].as_bool().unwrap_or(false);
    let (mut body, tool_map) = match protocol::convert_request(&body, source, target, &resolved.model.model_id) {
        Ok(converted) => converted,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, Json(source.error(&error.to_string()))).into_response(),
    };
    if streaming && target == Protocol::Chat {
        if !body["stream_options"].is_object() { body["stream_options"] = json!({}); }
        body["stream_options"]["include_usage"] = json!(true);
    }
    let url = endpoint_url(&resolved.provider.base_url, target.path());
    let mut request = context.client.post(url).json(&body)
        .header(header::ACCEPT, if streaming { "text/event-stream" } else { "application/json" });
    if target == Protocol::Messages {
        request = request.header("anthropic-version", "2023-06-01");
    }
    if resolved.provider.kind != ProviderKind::Ollama {
        let account = format!("provider:{}", resolved.provider.id);
        let Some(key) = context.store.read_secret(&account) else {
            return (StatusCode::PRECONDITION_REQUIRED, Json(source.error(&format!(
                "{} requires an API key. Add it in AutoJev → Providers.", resolved.provider.name
            )))).into_response();
        };
        request = if target == Protocol::Messages { request.header("x-api-key", key) } else { request.bearer_auth(key) };
    }
    if let Some(value) = headers.get("user-agent") { request = request.header("user-agent", value); }
    // Provider-specific beta/version headers apply only to unchanged protocols.
    if source == target {
        for name in ["anthropic-version", "anthropic-beta", "openai-beta"] {
            if let Some(value) = headers.get(name) { request = request.header(name, value); }
        }
    }
    if resolved.provider.kind == ProviderKind::Openrouter {
        request = request
            .header("HTTP-Referer", "https://autojev.ai")
            .header("X-Title", "AutoJev");
    }

    let upstream = match tokio::time::timeout(std::time::Duration::from_secs(config.gateway.response_timeout_seconds), request.send()).await {
        Ok(Ok(response)) => response,
        Err(_) => return error_response(StatusCode::GATEWAY_TIMEOUT, "Upstream response timed out"),
        Ok(Err(error)) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                &format!("Could not reach {}: {error}", resolved.provider.name),
            )
        }
    };
    let status = upstream.status();
    let response_headers = upstream.headers().clone();
    let success = status.is_success();
    capture.lock().unwrap().upstream(target, response_headers.get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok()).is_some_and(|v| v.starts_with("text/event-stream")));

    let request_metadata = capture.lock().unwrap().log.clone();
    let _ = context.store.add_event(RouteEvent {
        id: request_metadata.id,
        created_at: request_metadata.created_at,
        endpoint: endpoint.into(),
        provider_name: resolved.provider.name.clone(),
        model_name: resolved.model.name.clone(),
        reason: resolved.decision.reason.clone(),
        source: resolved.decision.source.clone(),
        estimated_input_tokens: input.estimated_context_tokens,
        estimated_cost: resolved.decision.estimated_cost,
        estimated_savings: resolved.decision.estimated_savings,
        success,
    });

    let mut builder = Response::builder().status(status);
    for name in [header::CONTENT_TYPE, header::CACHE_CONTROL, header::RETRY_AFTER] {
        if let Some(value) = response_headers.get(&name) {
            builder = builder.header(name, value);
        }
    }
    for name in ["x-request-id", "openai-request-id", "request-id"] {
        if let Some(value) = response_headers.get(name) {
            builder = builder.header(name, value);
        }
    }
    builder = builder
        .header("x-autojev-model", &resolved.model.model_id)
        .header("x-autojev-route-source", &resolved.decision.source);
    if source != target {
        if !success {
            let error = read_upstream_json(upstream, capture.clone(), config.gateway.stream_idle_seconds).await.ok();
            let message = error.as_ref().and_then(|body| body.pointer("/error/message").and_then(Value::as_str))
                .unwrap_or("Upstream rejected the converted request");
            return builder.header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(source.error(message).to_string())).unwrap();
        }
        if streaming {
            let is_sse = response_headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with("text/event-stream"));
            if !is_sse {
                return (StatusCode::BAD_GATEWAY, Json(source.error("Upstream did not return the requested event stream"))).into_response();
            }
            return builder.header(header::CONTENT_TYPE, "text/event-stream")
                .header(header::CACHE_CONTROL, "no-cache")
                .body(Body::from_stream(protocol::converted_stream_observed(crate::traffic::observe(timed_stream(upstream.bytes_stream(), config.gateway.stream_idle_seconds), capture.clone()), target, source, resolved.model.model_id, tool_map, { let capture = capture.clone(); move || capture.lock().unwrap().log.error = "Response conversion failed".into() }))).unwrap();
        }
        let converted = match read_upstream_json(upstream, capture.clone(), config.gateway.stream_idle_seconds).await.and_then(|body|
            protocol::convert_response(&body, target, source, &resolved.model.model_id, &tool_map)) {
            Ok(body) => body,
            Err(error) => return (StatusCode::BAD_GATEWAY, Json(source.error(&error.to_string()))).into_response(),
        };
        return builder.header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(converted.to_string())).unwrap();
    }
    if !streaming && success {
        return match read_upstream_json(upstream,capture.clone(),config.gateway.stream_idle_seconds).await {
            Ok(body)=>{
                if body.get("error").is_some_and(|v|!v.is_null()) {
                    let code=body.pointer("/error/code").and_then(Value::as_u64).filter(|n|(400..600).contains(n)).unwrap_or(502) as u16;
                    builder=builder.status(StatusCode::from_u16(code).unwrap());
                }
                builder.header(header::CONTENT_TYPE,"application/json").body(Body::from(body.to_string())).unwrap()
            },
            Err(_)=>error_response(StatusCode::BAD_GATEWAY,"Upstream returned invalid JSON or the response body timed out")
        };
    }
    let stream = crate::traffic::observe(timed_stream(upstream.bytes_stream(), config.gateway.stream_idle_seconds), capture).map(|chunk| chunk.map_err(std::io::Error::other));
    builder.body(Body::from_stream(stream)).unwrap_or_else(|_| {
        error_response(StatusCode::INTERNAL_SERVER_ERROR, "Could not construct upstream response")
    })
}

async fn read_upstream_json(response: reqwest::Response, capture: crate::traffic::SharedCapture, idle: u64) -> anyhow::Result<Value> {
    let mut body = Vec::new();
    let stream = crate::traffic::observe(response.bytes_stream(), capture.clone());
    futures_util::pin_mut!(stream);
    while let Some(chunk) = tokio::time::timeout(std::time::Duration::from_secs(idle), stream.next()).await? {
        let chunk = chunk?;
        if body.len() + chunk.len() > 16 * 1024 * 1024 { anyhow::bail!("Upstream response exceeds the 16 MiB conversion limit"); }
        body.extend_from_slice(&chunk);
    }
    capture.lock().unwrap().end_body();
    serde_json::from_slice(&body).map_err(|_| anyhow::anyhow!("Upstream returned invalid JSON"))

}

// Advisory estimate: do not tokenize image URLs/base64 as text. Image tokenization
// varies by provider/resolution; use a bounded 1024-token allowance per image.
fn estimate_request_tokens(body: &Value) -> u64 {
    fn units(value: &Value) -> u64 {
        match value {
            Value::Object(map) if matches!(map.get("type").and_then(Value::as_str), Some("image" | "image_url" | "input_image")) => 4096,
            Value::Object(map) => map.iter().map(|(key, value)| key.len() as u64 + units(value)).sum(),
            Value::Array(items) => items.iter().map(units).sum(),
            Value::String(text) => text.chars().map(|c| if c.is_ascii() { 1 } else { 4 }).sum(),
            _ => 4,
        }
    }
    ["messages", "input", "instructions", "system", "tools", "prompt"].iter()
        .filter_map(|key| body.get(key)).map(units).sum::<u64>().div_ceil(4).max(1)
}

fn inspect_request(body: &Value, endpoint: &str) -> RoutePreviewInput {

    let prompt = routing_prompt(body);
    let requires_tools = body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    let requires_vision = ["input", "messages"]
        .iter()
        .filter_map(|key| body.get(key))
        .any(has_image_content);
    RoutePreviewInput {
        prompt,
        endpoint: endpoint.into(),
        requires_tools,
        requires_vision,
        estimated_context_tokens: estimate_request_tokens(body),
        requested_model: body.get("model").and_then(Value::as_str).map(str::to_owned),
    }
}

// Inspect actual content parts, never words in prompts, instructions or tool schemas.
fn has_image_content(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(has_image_content),
        Value::Object(object) => {
            if matches!(object.get("type").and_then(Value::as_str),
                Some("input_image" | "image_url" | "image")) {
                return true;
            }
            object.get("content").is_some_and(has_image_content)
        }
        _ => false,
    }
}

// Classify the latest user task, not system prompts, tool schemas or old turns.
// Only this locally derived metadata is sent to the decision service.
fn routing_prompt(body: &Value) -> String {
    for key in ["messages", "input"] {
        if let Some(items) = body.get(key).and_then(Value::as_array) {
            for item in items.iter().rev() {
                if item["role"] != "user" { continue; }
                let mut text = String::new();
                let mut remaining = 4_000;
                collect_text(&item["content"], &mut text, &mut remaining, 0);
                // Anthropic tool results also have role=user; skip them.
                if !text.trim().is_empty() { return text; }
            }
        }
    }
    let mut text = String::new();
    let mut remaining = 4_000;
    if let Some(input) = body.get("input").filter(|v| v.is_string()) {
        collect_text(input, &mut text, &mut remaining, 0);
    } else if let Some(prompt) = body.get("prompt") {
        collect_text(prompt, &mut text, &mut remaining, 0);
    }
    text
}

fn collect_text(value: &Value, result: &mut String, remaining: &mut usize, depth: usize) {
    if depth > 8 || *remaining == 0 {
        return;
    }
    match value {
        Value::String(text) => {
            if !result.is_empty() {
                result.push(' ');
                *remaining -= 1;
            }
            for c in text.chars().take(*remaining) {
                result.push(c);
                *remaining -= 1;
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_text(item, result, remaining, depth + 1);
                if *remaining == 0 { break; }
            }
        }
        Value::Object(object) if matches!(object.get("type").and_then(Value::as_str), Some("text" | "input_text")) => {
            if let Some(text) = object.get("text") {
                collect_text(text, result, remaining, depth + 1);
            }
        }
        _ => {}
    }
}

fn session_id(headers: &HeaderMap, body: &Value) -> Option<String> {
    headers
        .get("x-autojev-session-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            body.pointer("/metadata/user_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            body.get("conversation")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            body.get("prompt_cache_key")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            body.get("previous_response_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn still_eligible(
    route: &ResolvedRoute,
    input: &RoutePreviewInput,
    config: &crate::config::AppConfig,
) -> bool {
    let speed_route = input.requested_model.as_deref().and_then(|id|id.strip_prefix("autojev/"))
        .and_then(|id|config.routes.iter().find(|r|r.id==id && r.strategy=="jev"))
        .is_some_and(|r|r.automatic_policy.as_ref().unwrap_or(&config.policy).decision_preference=="speed");
    !crate::router::balanced_session_is_slow(config,input,&route.model)
        && (!speed_route || crate::router::speed_capable(&route.model,input))
        && route.model.enabled
        && config.models.iter().any(|m| m.id == route.model.id && m.enabled && (!input.requires_vision || m.supports_vision))
        && crate::router::protocol_matches(&route.model, &route.provider, &input.endpoint)
        && config
            .providers
            .iter()
            .any(|provider| provider.id == route.provider.id && provider.enabled)
}

pub(crate) fn endpoint_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") && path.starts_with("/v1/") {
        format!("{}{}", base, &path[3..])
    } else {
        format!("{base}{path}")
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let payload = json!({
        "error": {
            "message": message,
            "type": "autojev_proxy_error",
            "code": status.as_u16()
        }
    });
    (status, Json(payload)).into_response()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn hosted_search_requires_native_responses_before_model_selection() {
        let config=crate::config::AppConfig::default();
        let mut model=config.models[0].clone();
        let provider=config.providers.iter().find(|p|p.id==model.provider_id).unwrap();
        let body=json!({"model":"autojev/fast","input":"hello","tools":[{"type":"web_search"}]});
        model.api_type="chat_completions".into();
        assert!(request_compatible(&body,Protocol::Responses,&model,provider).is_err());
        model.api_type="messages".into();
        assert!(request_compatible(&body,Protocol::Responses,&model,provider).is_err());
        model.api_type="responses".into();
        assert!(request_compatible(&body,Protocol::Responses,&model,provider).is_ok());
        model.api_type="chat_completions".into();
        assert!(request_compatible(&json!({"input":"hello"}),Protocol::Responses,&model,provider).is_ok());
    }

    #[test]
    fn image_payload_size_does_not_inflate_token_estimate() {
        for kind in ["image_url", "input_image", "image"] {
            let make = |data: String| json!({"messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":kind,"image_url":data.clone(),"source":{"type":"base64","data":data}}]}]});
            let small = inspect_request(&make("a".into()), "chat/completions");
            let large = inspect_request(&make("a".repeat(1_000_000)), "chat/completions");
            assert_eq!(small.estimated_context_tokens, large.estimated_context_tokens);
            assert!(large.estimated_context_tokens >= 1024 && large.estimated_context_tokens < 1100);
            assert!(large.requires_vision);
        }
        assert!(estimate_request_tokens(&json!({"messages":[{"role":"user","content":"a".repeat(40_000)}]})) >= 10000);
    }

    #[tokio::test]
    async fn image_request_invalidates_text_only_session_model() {
        let mut config = crate::config::AppConfig::default();
        config.models[0].supports_vision = false;
        let mut input = inspect_request(&json!({"model":format!("autojev/model/{}",config.models[0].id),"messages":[{"role":"user","content":"hello"}]}), "chat/completions");
        let route = decide(&config, &input, &Client::new(), None).await.unwrap();
        assert!(still_eligible(&route, &input, &config));
        input.requires_vision = true;
        assert!(!still_eligible(&route, &input, &config));
        config.models[0].supports_vision = true;
        assert!(still_eligible(&route, &input, &config));
    }

    #[test]
    fn routing_reads_latest_user_task_in_each_protocol() {
        for endpoint in ["chat/completions", "messages", "responses"] {
            let key = if endpoint == "responses" { "input" } else { "messages" };
            let body = json!({key: [
                {"role":"system", "content":"security migration architecture"},
                {"role":"user", "content":"Investigate production authentication"},
                {"role":"assistant", "content":"security vulnerabilities"},
                {"role":"user", "content":[{"type":"text","text":"翻译"}, {"type":"text","text":"你好"}]},
                {"role":"user", "content":[{"type":"tool_result","content":"migration delete payment"}]},
                {"role":"tool", "content":"security incident"}
            ]});
            assert_eq!(inspect_request(&body, endpoint).prompt, "翻译 你好");
        }
        assert_eq!(inspect_request(&json!({"input":"Hi"}), "responses").prompt, "Hi");
        let body = json!({"messages":[{"role":"user","content":"界".repeat(10_000)}]});
        assert_eq!(inspect_request(&body, "messages").prompt.chars().count(), 4_000);
    }

    #[test]
    fn text_and_tool_schemas_do_not_require_vision() {
        let body = json!({
            "model": "autojev/auto",
            "instructions": "Use input_image and image_url when working with images.",
            "tools": [{"type":"function", "name":"view_image", "description":"Return input_image", "parameters":{"properties":{"image_url":{"type":"string"}}}}],
            "input": [{"role":"user", "content":[{"type":"input_text", "text":"Explain image_url and input_image"}]}]
        });
        let input = inspect_request(&body, "responses");
        assert!(!input.requires_vision);
        assert!(input.requires_tools);
    }

    #[test]
    fn actual_image_parts_require_vision_in_all_supported_formats() {
        for (endpoint, body) in [
            ("responses", json!({"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.png"}]}]})),
            ("chat/completions", json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]})),
            ("messages", json!({"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","data":"abc"}}]}]})),
            ("messages", json!({"messages":[{"role":"user","content":[{"type":"tool_result","content":[{"type":"image","source":{"type":"base64","data":"abc"}}]}]}]})),
        ] { assert!(inspect_request(&body, endpoint).requires_vision, "{endpoint}"); }
    }

    #[test]
    fn builds_provider_urls_without_duplicate_v1() {
        assert_eq!(
            endpoint_url("https://example.com/v1", "/v1/responses"),
            "https://example.com/v1/responses"
        );
        assert_eq!(
            endpoint_url("https://openrouter.ai/api", "/v1/messages"),
            "https://openrouter.ai/api/v1/messages"
        );
    }

    #[tokio::test]
    async fn serves_health_on_loopback() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(ConfigStore::load(directory.path().join("autojev.db")).unwrap());
        let reserved=TcpListener::bind("127.0.0.1:0").await.unwrap();let port=reserved.local_addr().unwrap().port();drop(reserved);
        store.update(|config| config.port = port).unwrap();
        let handle = start(store.clone()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let response: Value = reqwest::get(format!("http://127.0.0.1:{port}/health"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(response["status"], "ok");
        let catalog:Value=reqwest::get(format!("http://127.0.0.1:{port}/v1/models")).await.unwrap().json().await.unwrap();
        assert_eq!(catalog["object"],"list");
        // Public IDs are <provider>/<model> for models and autojev/<route> for routes; each must resolve back.
        let config=store.read();let ids=catalog["data"].as_array().unwrap();assert!(!ids.is_empty());
        assert!(ids.iter().all(|v|crate::router::normalize_requested_model(&config,v["id"].as_str()).unwrap().is_some_and(|m|m.starts_with("autojev/"))));
        handle.stop().await;
        assert!(TcpListener::bind(("127.0.0.1",port)).await.is_ok());
    }
}

#[cfg(test)]
mod route_forward_tests {
    use super::*;
    #[tokio::test]
    async fn load_balancing_fails_over_on_rate_limit() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let calls = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let observed = calls.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
                let calls = observed.clone();
                async move {
                    let model = body["model"].as_str().unwrap().to_owned();
                    calls.lock().unwrap().push(model.clone());
                    if model == "primary" { (StatusCode::TOO_MANY_REQUESTS, Json(json!({"error":{"message":"busy"}}))).into_response() }
                    else {Json(json!({"model":model,"choices":[]})).into_response()}
                }
            }))).await.unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ConfigStore::load(dir.path().join("test.db")).unwrap());
        store.update(|config| {
            config.models.truncate(1);
            config.providers[0].base_url = format!("http://127.0.0.1:{port}");
            config.models[0].model_id = "primary".into();
            let mut backup = config.models[0].clone(); backup.id = "backup".into(); backup.model_id = "backup".into();
            config.models.push(backup);
            config.routes = vec![crate::config::RouteRule { all_models: false, automatic_policy: None,id:"fallback".into(),name:"Fallback".into(),strategy:"round_robin".into(),enabled:true,
                model_ids:config.models.iter().map(|m|m.id.clone()).collect(),
                model_settings: std::collections::HashMap::from([(config.models[0].id.clone(),crate::config::RouteModelSettings{priority:10,weight:1})])}];
        }).unwrap();
        store.write_secret("provider:openrouter", "test-only-key").unwrap();
        let context = ProxyContext {store,client:Client::new(),sessions:Default::default(),health:Default::default()};
        let response = forward(context,HeaderMap::new(),json!({"model":"autojev/fallback","messages":[{"role":"user","content":"hello"}]}),"/v1/chat/completions","chat/completions").await;
        assert_eq!(response.status(),StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(),4096).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["model"],"backup");
        assert_eq!(*calls.lock().unwrap(),vec!["primary","backup"]);
        server.abort();
    }
    #[tokio::test]
    async fn hermes_switches_models_in_same_session_and_rejects_unselected_ids() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let upstream = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/chat/completions", post(|headers: HeaderMap, Json(body): Json<Value>| async move {
                assert!(headers.get("x-autojev-agent").is_none());
                Json(json!({"model":body["model"],"choices":[]}))
            }))).await.unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ConfigStore::load(dir.path().join("test.db")).unwrap());
        store.update(|config| {
            config.models.truncate(1);
            config.providers[0].base_url = format!("http://127.0.0.1:{port}");
            config.models[0].model_id = "first".into();
            let mut second = config.models[0].clone();
            second.id = "second-id".into(); second.model_id = "second".into();
            config.models.push(second);
            let bindings = config.models.iter().map(|m| format!("model/{}",m.id)).collect::<Vec<_>>();
            config.agent_catalogs.insert("hermes".into(), crate::agent_catalog::build(config, &bindings).unwrap());
        }).unwrap();
        store.write_secret("provider:openrouter", "test-only-key").unwrap();
        let context = ProxyContext {store:store.clone(),client:Client::new(),sessions:Default::default(),health:Default::default()};
        let mut headers = HeaderMap::new();
        headers.insert("x-autojev-agent", HeaderValue::from_static("hermes"));
        headers.insert("x-autojev-session-id", HeaderValue::from_static("same-session"));
        let provider = store.read().models[0].provider_id.clone();
        for (public, upstream) in [(format!("{provider}/first"), "first"), (format!("{provider}/second"), "second"), ("not-selected".into(), "")] {
            let body = json!({"model":public,"messages":[{"role":"user","content":"hello"}]});
            let response = forward(context.clone(), headers.clone(), body, "/v1/chat/completions", "chat/completions").await;
            if upstream.is_empty() {assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);}
            else {
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
                assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["model"], upstream);
            }
        }
        upstream.abort();
    }
    #[tokio::test]
    async fn forwards_named_route_and_rejects_disabled_route_in_same_session() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let upstream = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/chat/completions", post(
                |headers: HeaderMap, Json(body): Json<Value>| async move {
                    assert_eq!(headers.get("authorization").unwrap(), "Bearer test-only-key");
                    assert!(headers.get("x-autojev-binding").is_none());
                    Json(json!({"model": body["model"], "choices": []}))
                }))).await.unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.db");
        let store = Arc::new(ConfigStore::load(path.clone()).unwrap());
        store.update(|config| {
            config.providers[0].base_url = format!("http://127.0.0.1:{port}");
            config.models.truncate(1);
            config.routes = vec![crate::config::RouteRule { all_models: false, automatic_policy: None, model_settings: Default::default(), id: "daily".into(), name: "Daily".into(),
                enabled: true, strategy: "fixed".into(), model_ids: vec![config.models[0].id.clone()] }];
        }).unwrap();
        store.write_secret("provider:openrouter", "test-only-key").unwrap();
        assert_eq!(ConfigStore::load(path).unwrap().read().routes[0].id, "daily");
        let context = ProxyContext { store: store.clone(), client: Client::new(), sessions: Default::default(), health: Default::default() };
        let body = json!({"model":"public-model-id", "messages":[{"role":"user","content":"hello"}]});
        let mut headers = HeaderMap::new();
        headers.insert("x-autojev-binding", HeaderValue::from_static("daily"));
        headers.insert("x-autojev-session-id", HeaderValue::from_static("test-session"));
        let response = forward(context.clone(), headers.clone(), body.clone(), "/v1/chat/completions", "chat/completions").await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let payload: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["model"], store.read().models[0].model_id);
        assert_eq!(store.request_logs("").unwrap()[0].status, "success");
        store.update(|config| config.routes[0].enabled = false).unwrap();
        let response = forward(context, headers, body, "/v1/chat/completions", "chat/completions").await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let logs = store.request_logs("").unwrap();
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].status, "error");
        assert_eq!(logs[0].status_code, 422);
        assert_eq!(logs[0].input_tokens, None);
        upstream.abort();
    }
}

#[cfg(test)]
mod protocol_forward_tests {
    use super::*;
    use crate::protocol::tests::{request, response, wire};

    #[tokio::test]
    async fn six_directions_use_upstream_path_and_auth_for_json_and_streams() {
        let protocols = [Protocol::Chat, Protocol::Responses, Protocol::Messages];
        for source in protocols {
            for target in protocols {
                if source == target { continue; }
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let port = listener.local_addr().unwrap().port();
                let server = tokio::spawn(async move {
                    axum::serve(listener, Router::new().route(target.path(), post(move |headers: HeaderMap, Json(body): Json<Value>| async move {
                        assert_eq!(body["model"], "test-upstream-model");
                        assert!(body.to_string().contains("call_1"));
                        assert!(headers.get("anthropic-beta").is_none());
                        if target == Protocol::Messages {
                            assert_eq!(headers["x-api-key"], "test-only-key");
                            assert!(headers.get("authorization").is_none());
                            assert_eq!(headers["anthropic-version"], "2023-06-01");
                            assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "call_1");
                        } else {
                            assert_eq!(headers["authorization"], "Bearer test-only-key");
                            assert!(headers.get("x-api-key").is_none());
                            assert!(headers.get("anthropic-version").is_none());
                        }
                        if body["stream"] == true {
                            let chunks = wire(target).as_bytes().chunks(7).map(|bytes| Ok::<_, std::io::Error>(axum::body::Bytes::copy_from_slice(bytes))).collect::<Vec<_>>();
                            ([("content-type", "text/event-stream")], Body::from_stream(futures_util::stream::iter(chunks))).into_response()
                        } else { Json(response(target)).into_response() }
                    }))).await.unwrap();
                });
                let dir = tempfile::tempdir().unwrap();
                let store = Arc::new(ConfigStore::load(dir.path().join("protocol.db")).unwrap());
                store.update(|config| {
                    config.models.truncate(1);
                    config.models[0].model_id = "test-upstream-model".into();
                    config.models[0].api_type = target.path().trim_start_matches("/v1/").into();
                    config.providers[0].base_url = format!("http://127.0.0.1:{port}/v1");
                    config.providers[0].kind = ProviderKind::OpenaiCompatible;
                    config.policy.use_jev_when_ambiguous = false;
                }).unwrap();
                store.write_secret("provider:openrouter", "test-only-key").unwrap();
                let context = ProxyContext { store, client: Client::new(), sessions: Default::default(), health: Default::default() };
                for streaming in [false, true] {
                    let mut body = request(source, streaming);
                    body["model"] = "autojev/auto".into();
                    let mut headers = HeaderMap::new();
                    headers.insert("anthropic-beta", HeaderValue::from_static("client-only-feature"));
                    let reply = forward(context.clone(), headers, body, source.path(), source.path().trim_start_matches("/v1/")).await;
                    assert_eq!(reply.status(), StatusCode::OK, "{source:?} → {target:?}");
                    assert_eq!(reply.headers()["x-autojev-model"], "test-upstream-model");
                    let content_type = reply.headers()[header::CONTENT_TYPE].to_str().unwrap().to_owned();
                    let bytes = axum::body::to_bytes(reply.into_body(), 64 * 1024).await.unwrap();
                    if streaming {
                        assert!(content_type.starts_with("text/event-stream"));
                        let text = String::from_utf8(bytes.to_vec()).unwrap();
                        assert!(text.contains(match source { Protocol::Chat => "[DONE]", Protocol::Responses => "response.completed", Protocol::Messages => "message_stop" }), "{text}");
                    } else {
                        let body: Value = serde_json::from_slice(&bytes).unwrap();
                        assert!(body.to_string().contains("call_2"));
                        assert!(body.to_string().contains("你好"));
                        assert!(content_type.starts_with("application/json"));
                    }
                }
                let logs = context.store.request_logs("").unwrap();
                assert_eq!(logs.len(), 2);
                for log in logs {
                    assert_eq!(log.status, "success", "{source:?} → {target:?}: {}", log.error);
                    assert_eq!(log.input_tokens, Some(12));
                    assert_eq!(log.output_tokens, Some(7));
                    assert!(log.estimated_cost.is_some());
                }
                server.abort();
            }
        }
    }

    #[tokio::test]
    async fn preserves_upstream_http_errors_in_client_protocol() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/chat/completions", post(|| async {
                (StatusCode::TOO_MANY_REQUESTS, Json(json!({"error":{"message":"rate limited"}})))
            }))).await.unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ConfigStore::load(dir.path().join("errors.db")).unwrap());
        store.update(|config| { config.models.truncate(1); config.providers[0].base_url = format!("http://127.0.0.1:{port}"); }).unwrap();
        store.write_secret("provider:openrouter", "test-only-key").unwrap();
        let mut body = request(Protocol::Messages, false);
        body["model"] = "auto".into();
        let reply = forward(ProxyContext { store, client: Client::new(), sessions: Default::default(), health: Default::default() }, HeaderMap::new(), body, "/v1/messages", "messages").await;
        assert_eq!(reply.status(), StatusCode::TOO_MANY_REQUESTS);
        let body: Value = serde_json::from_slice(&axum::body::to_bytes(reply.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["message"], "rate limited");
        server.abort();
    }
}

fn timed_stream<S,E>(stream:S,seconds:u64)->impl futures_util::Stream<Item=Result<axum::body::Bytes,std::io::Error>>+Send
where S:futures_util::Stream<Item=Result<axum::body::Bytes,E>>+Send,E:std::error::Error+Send+Sync+'static {
    futures_util::stream::unfold((Box::pin(stream),false),move |(mut stream,done)|async move{
        if done{return None;}
        match tokio::time::timeout(std::time::Duration::from_secs(seconds),stream.next()).await {
            Ok(Some(chunk))=>{let failed=chunk.is_err();Some((chunk.map_err(std::io::Error::other),(stream,failed)))},
            Ok(None)=>None,
            Err(_)=>Some((Err(std::io::Error::new(std::io::ErrorKind::TimedOut,"Upstream idle timeout")),(stream,true))),
        }
    })
}
fn observe_health(response:Response,lease:crate::resilience::Lease,settings:crate::resilience::Settings,capture:crate::traffic::SharedCapture)->Response {
    let(parts,body)=response.into_parts();
    let stream=futures_util::stream::unfold((body.into_data_stream(),Some(lease),settings,capture),|(mut stream,mut lease,settings,capture)|async move{
        match stream.next().await{
            Some(chunk)=>{if chunk.is_err(){if let Some(l)=lease.take(){l.complete(502,None,&settings);}}Some((chunk,(stream,lease,settings,capture)))},
            None=>{let failed=capture.lock().unwrap().failed_body();if let Some(l)=lease.take(){l.complete(if failed{502}else{200},None,&settings);}None}
        }
    });Response::from_parts(parts,Body::from_stream(stream))
}

async fn model_catalog(State(context):State<ProxyContext>,headers:HeaderMap)->Response {
    let config=context.store.read();
    if let Some(agent)=headers.get("x-autojev-agent").and_then(|v|v.to_str().ok()) {
        let entries=config.agent_catalogs.get(agent).cloned().unwrap_or_default();
        return Json(json!({"object":"list","data":entries.iter().map(|e|json!({"id":e.id,"name":e.name,"object":"model","created":0,"owned_by":"autojev"})).collect::<Vec<_>>()})).into_response();
    }
    let bindings=config.models.iter().filter(|m|m.enabled&&config.providers.iter().any(|p|p.id==m.provider_id&&p.enabled)).map(|m|format!("model/{}",m.id))
        .chain(config.routes.iter().filter(|r|r.enabled).map(|r|r.id.clone())).collect::<Vec<_>>();
    let data=bindings.iter().filter_map(|b|crate::agent_catalog::build(&config,std::slice::from_ref(b)).ok()).flatten()
        .map(|e|json!({"id":e.id,"name":e.name,"object":"model","created":0,"owned_by":"autojev"})).collect::<Vec<_>>();
    Json(json!({"object":"list","data":data})).into_response()
}

#[cfg(test)]
mod availability_tests {
    use super::*;
    #[tokio::test]
    async fn retry_matrix_limits_and_cooldown() {
        for (status,max,expected) in [(400,4,1),(401,4,2),(402,4,2),(403,4,2),(401,1,1),(422,4,1),(429,4,2),(500,4,2),(503,4,2),(429,1,1)] {
            let listener=TcpListener::bind("127.0.0.1:0").await.unwrap();let port=listener.local_addr().unwrap().port();
            let calls=Arc::new(std::sync::atomic::AtomicUsize::new(0));let observed=calls.clone();
            let server=tokio::spawn(async move{axum::serve(listener,Router::new().route("/v1/chat/completions",post(move|Json(body):Json<Value>|{let calls=observed.clone();async move{
                calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
                if body["model"]=="primary" {(StatusCode::from_u16(status).unwrap(),[("retry-after","60")],Json(json!({"error":{"message":"test failure"}}))).into_response()}
                else{Json(json!({"choices":[{"message":{"content":"ok"}}]})).into_response()}
            }}))).await.unwrap();});
            let dir=tempfile::tempdir().unwrap();let store=Arc::new(ConfigStore::load(dir.path().join("test.db")).unwrap());
            store.update(|c|{c.models.truncate(1);c.models[0].model_id="primary".into();c.providers[0].base_url=format!("http://127.0.0.1:{port}");c.gateway.max_attempts=max;
                let mut backup=c.models[0].clone();backup.id="backup".into();backup.model_id="backup".into();c.models.push(backup);
                c.routes=vec![crate::config::RouteRule{ all_models: false,id:"test".into(),name:"test".into(),enabled:true,strategy:"round_robin".into(),automatic_policy:None,model_ids:c.models.iter().map(|m|m.id.clone()).collect(),model_settings:HashMap::from([(c.models[0].id.clone(),crate::config::RouteModelSettings{priority:10,weight:1})])}];}).unwrap();
            store.write_secret("provider:openrouter","test").unwrap();let primary=store.read().models[0].id.clone();
            let context=ProxyContext{store:store.clone(),client:Client::new(),sessions:Default::default(),health:Default::default()};
            let response=forward(context.clone(),HeaderMap::new(),json!({"model":"autojev/test","messages":[]}),"/v1/chat/completions","chat/completions").await;
            assert_eq!(response.status().as_u16(),if expected==2{200}else{status});
            axum::body::to_bytes(response.into_body(),4096).await.unwrap();
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),expected);
            let logs=store.request_logs("").unwrap();assert_eq!(logs[0].attempts.len(),expected);
            if expected==2{assert_eq!(logs[0].status,"success");assert!(logs[0].error.is_empty());}
            if matches!(status,401|402|403|429) && expected==2{
                assert!(!context.health.available(&primary));
                let again=forward(context.clone(),HeaderMap::new(),json!({"model":"autojev/test","messages":[]}),"/v1/chat/completions","chat/completions").await;
                assert_eq!(again.status(),StatusCode::OK);axum::body::to_bytes(again.into_body(),4096).await.unwrap();
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),expected+1);
                context.health.acquire("backup").unwrap().complete(429,None,&store.read().gateway);
                let unavailable=forward(context.clone(),HeaderMap::new(),json!({"model":"autojev/test","messages":[]}),"/v1/chat/completions","chat/completions").await;
                assert_eq!(unavailable.status(),StatusCode::SERVICE_UNAVAILABLE);assert!(unavailable.headers().contains_key("retry-after"));
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),expected+1);
            }
            server.abort();
        }
    }
    #[tokio::test]
    async fn idle_timeout_terminates_stream_without_replay(){
        let stream=futures_util::stream::pending::<Result<axum::body::Bytes,std::io::Error>>();
        let timed=timed_stream(stream,1);futures_util::pin_mut!(timed);
        assert_eq!(timed.next().await.unwrap().unwrap_err().kind(),std::io::ErrorKind::TimedOut);
        assert!(timed.next().await.is_none());
    }
    #[tokio::test]
    async fn successful_headers_do_not_mark_failed_body_healthy(){
        let h=crate::resilience::Health::default();let mut settings=crate::resilience::Settings::default();settings.failure_threshold=1;
        let lease=h.acquire("model").unwrap();let capture=crate::traffic::Capture::new("chat/completions",&json!({}),&HeaderMap::new());
        let response=Response::new(Body::from_stream(futures_util::stream::iter([Err::<axum::body::Bytes,_>(std::io::Error::other("disconnected"))])));
        let response=observe_health(response,lease,settings,capture);
        assert!(axum::body::to_bytes(response.into_body(),1024).await.is_err());assert!(!h.available("model"));
    }
}

#[cfg(test)]
mod pause_tests {
    use super::*;
    #[tokio::test]
    async fn pause_retains_listener_state_and_returns_not_found_with_retry_hint() {
        let task = tokio::spawn(std::future::pending::<()>());
        let handle = ProxyHandle { shutdown: None, health: Default::default(), task, paused: Arc::new(std::sync::atomic::AtomicBool::new(false)) };
        handle.set_paused(true);
        assert!(handle.running());
        assert!(handle.paused());
        let response = paused_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(response.headers()["x-should-retry"], "false");
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "gateway_paused");
        assert_eq!(body["error"]["type"], "not_found_error");
        handle.set_paused(false);
        assert!(!handle.paused());
        assert!(handle.running());
        handle.task.abort();
    }
}
