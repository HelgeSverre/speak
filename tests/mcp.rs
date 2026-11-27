//! `speak --mcp`: the Model Context Protocol server on stdio.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Server {
    fn start() -> Server {
        let mut child = Command::new(env!("CARGO_BIN_EXE_speak"))
            .arg("--mcp")
            .env("SPEAK_LIGHTS", "0")
            .env("SPEAK_DATA_DIR", "/nonexistent")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Server {
            child,
            stdin,
            stdout,
        }
    }

    fn send(&mut self, msg: Value) {
        writeln!(self.stdin, "{msg}").unwrap();
    }

    fn request(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.send(json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let reply: Value = serde_json::from_str(&line).expect("every stdout line is JSON");
        assert_eq!(reply["jsonrpc"], "2.0");
        assert_eq!(reply["id"], id, "{reply}");
        reply
    }

    fn call(&mut self, id: u64, tool: &str, arguments: Value) -> Value {
        let reply = self.request(
            id,
            "tools/call",
            json!({ "name": tool, "arguments": arguments }),
        );
        reply["result"].clone()
    }
}

fn text(result: &Value) -> &str {
    result["content"][0]["text"].as_str().unwrap()
}

#[test]
fn initialize_list_and_speak_to_a_file() {
    let wav = std::env::temp_dir().join(format!("speak-mcp-{}.wav", std::process::id()));
    let _ = std::fs::remove_file(&wav);
    let mut server = Server::start();

    let init = server.request(
        1,
        "initialize",
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0" },
        }),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "speak");
    assert!(init["result"]["capabilities"]["tools"].is_object());
    server.send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));

    let list = server.request(2, "tools/list", json!({}));
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["speak", "list_voices"]);
    assert!(
        list["result"]["tools"][0]["inputSchema"]["properties"]["sing"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("daisy"))
    );

    let voices = server.call(3, "list_voices", json!({}));
    assert_eq!(voices["isError"], false);
    assert!(text(&voices).contains("klatt-female"));

    let spoken = server.call(
        4,
        "speak",
        json!({ "text": "Hello from the server.", "voice": "klatt-pip", "rate": 220, "file": wav }),
    );
    assert_eq!(spoken["isError"], false, "{spoken}");
    assert!(text(&spoken).starts_with("Wrote ") && text(&spoken).ends_with(".wav"));
    let audio = std::fs::read(&wav).unwrap();
    assert_eq!(&audio[..4], b"RIFF");
    assert!(audio.len() > 10_000, "{} bytes", audio.len());

    // Mistakes come back as tool errors the agent can read.
    for (id, args, needle) in [
        (5, json!({}), "give text"),
        (
            6,
            json!({ "text": "hi", "voice": "nobody" }),
            "unknown voice",
        ),
        (7, json!({ "text": "hi", "file": "-" }), "stdout"),
        (8, json!({ "text": "hi", "volume": 2 }), "unknown field"),
    ] {
        let r = server.call(id, "speak", args);
        assert_eq!(r["isError"], true, "{r}");
        assert!(text(&r).contains(needle), "{r}");
    }
    let unknown = server.request(9, "tools/call", json!({ "name": "shout" }));
    assert_eq!(unknown["error"]["code"], -32602);
    let missing = server.request(10, "resources/list", json!({}));
    assert_eq!(missing["error"]["code"], -32601);

    // Closing stdin ends the server.
    drop(server.stdin);
    assert!(server.child.wait().unwrap().success());
    let _ = std::fs::remove_file(&wav);
}
