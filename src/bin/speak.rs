use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;

fn main() -> ExitCode {
    let start = Instant::now();
    let args = speak::cli::args::Args::parse();
    #[cfg(feature = "mcp")]
    if args.mcp {
        return exit(speak::cli::mcp::serve());
    }
    exit(speak::cli::pipeline::run(args, start))
}

fn exit(result: speak::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if !matches!(e, speak::Error::Interrupted) {
                eprintln!("speak: {e}");
            }
            ExitCode::from(e.exit_code() as u8)
        }
    }
}
