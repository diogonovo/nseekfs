use log::LevelFilter;
use std::env;

pub fn init_logging() {
    let default_level = LevelFilter::Off;

    let log_level = env::var("NSEEK_LOG")
        .ok()
        .and_then(|level| level.parse::<LevelFilter>().ok())
        .unwrap_or(default_level);

    let _ = env_logger::Builder::new()
        .format(|buf, record| {
            use std::io::Write;
            writeln!(
                buf,
                "[{} {}] {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter_level(log_level)
        .try_init();
}
