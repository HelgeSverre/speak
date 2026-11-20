use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;

fn main() -> ExitCode {
    let start = Instant::now();
    let args = speak::cli::args::Args::parse();
    match speak::cli::pipeline::run(args, start) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if !matches!(e, speak::Error::Interrupted) {
                eprintln!("speak: {e}");
            }
            ExitCode::from(e.exit_code() as u8)
        }
    }
}
