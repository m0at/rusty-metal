use std::fs;
use std::path::Path;
use std::process;

const AGENT_MD: &str = include_str!("../content/metal-agent.md");
const HINTS_MD: &str = include_str!("../content/metal-kernel-hints.md");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let force = args.iter().any(|a| a == "--force" || a == "-f");

    match args.get(1).map(String::as_str) {
        Some("init") => init(force),
        Some("check") => check(),
        Some("--version" | "-V") => println!("rusty-metal {}", env!("CARGO_PKG_VERSION")),
        Some("--help" | "-h") | None => usage(),
        Some(cmd) => {
            eprintln!("error: unknown command '{cmd}'");
            eprintln!();
            usage();
            process::exit(1);
        }
    }
}

fn usage() {
    eprintln!(
        "\
rusty-metal — Metal GPU agent for Claude Code

USAGE:
    rusty-metal init [--force]    Install .claude/agents/ with Metal agent + kernel hints
    rusty-metal check             Show what exists / what would be written
    rusty-metal --version         Print version

INSTALL:
    cargo install --path .        Build and install from local checkout
    cargo install rusty-metal     Install from crates.io (when published)"
    );
}

fn init(force: bool) {
    let agents_dir = Path::new(".claude/agents");

    if !Path::new(".git").exists() && !Path::new("CLAUDE.md").exists() {
        eprintln!("warning: no .git or CLAUDE.md found — are you in a project root?");
    }

    fs::create_dir_all(agents_dir).unwrap_or_else(|e| {
        eprintln!("error: failed to create {}: {e}", agents_dir.display());
        process::exit(1);
    });

    let wrote_agent = write_file(&agents_dir.join("metal.md"), AGENT_MD, force);
    let wrote_hints = write_file(&agents_dir.join("metal-kernel-hints.md"), HINTS_MD, force);

    if wrote_agent || wrote_hints {
        println!();
        println!("Metal agent ready. Use with: claude --agent metal");
    } else {
        println!();
        println!("Nothing changed. Use --force to overwrite existing files.");
    }
}

fn check() {
    let files = [
        ".claude/agents/metal.md",
        ".claude/agents/metal-kernel-hints.md",
    ];

    for path in &files {
        let p = Path::new(path);
        if p.exists() {
            let meta = fs::metadata(p).unwrap();
            let size = meta.len();
            println!("  exists  {path} ({size} bytes)");
        } else {
            println!("  missing {path}");
        }
    }
}

fn write_file(path: &Path, content: &str, force: bool) -> bool {
    let exists = path.exists();

    if exists && !force {
        println!("  skip    {} (exists)", path.display());
        return false;
    }

    fs::write(path, content).unwrap_or_else(|e| {
        eprintln!("error: failed to write {}: {e}", path.display());
        process::exit(1);
    });

    let verb = if exists { "updated" } else { "created" };
    println!("  {verb}  {}", path.display());
    true
}
