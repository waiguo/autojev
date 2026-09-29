fn main() {
    if autojev_lib::headless::entry() {
        return;
    }
    autojev_lib::run();
}
