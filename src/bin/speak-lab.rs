use std::process::ExitCode;

use clap::Parser;

fn main() -> ExitCode {
    let args = speak::cli::lab::LabArgs::parse();
    match speak::cli::lab::run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("speak-lab: {e}");
            ExitCode::from(e.exit_code() as u8)
        }
    }
}
