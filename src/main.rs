use std::fs;
use std::path::Path;
use std::process;

const AGENT_MD: &str = include_str!("../content/metal-agent.md");
const HINTS_MD: &str = include_str!("../content/metal-kernel-hints.md");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let force = args.iter().any(|a| a == "--force" || a == "-f");

    match args.get(1).map(String::as_str) {
        Some("init") => init(Path::new("."), force),
        Some("check") => check(Path::new(".")),
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

fn init(base_dir: &Path, force: bool) {
    let agents_dir = base_dir.join(".claude/agents");

    if !base_dir.join(".git").exists() && !base_dir.join("CLAUDE.md").exists() {
        eprintln!("warning: no .git or CLAUDE.md found — are you in a project root?");
    }

    fs::create_dir_all(&agents_dir).unwrap_or_else(|e| {
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

fn check(base_dir: &Path) {
    let files = [
        ".claude/agents/metal.md",
        ".claude/agents/metal-kernel-hints.md",
    ];

    for rel_path in &files {
        let p = base_dir.join(rel_path);
        if p.exists() {
            let meta = fs::metadata(&p).unwrap();
            let size = meta.len();
            println!("  exists  {rel_path} ({size} bytes)");
        } else {
            println!("  missing {rel_path}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a unique temp directory for a test. Caller is responsible for cleanup.
    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join("rusty-metal-tests")
            .join(name)
            .join(format!("{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ── 1. Content integrity ──────────────────────────────────────────

    #[test]
    fn agent_md_is_not_empty() {
        assert!(AGENT_MD.len() > 1000, "metal-agent.md should be substantial");
    }

    #[test]
    fn hints_md_is_not_empty() {
        assert!(HINTS_MD.len() > 1000, "metal-kernel-hints.md should be substantial");
    }

    #[test]
    fn agent_md_has_required_sections() {
        let required = [
            "# Metal Compute Agent",
            "## Architecture",
            "## Dispatch conventions",
            "## Performance rules",
            "## When NOT to use Metal GPU",
            "### System health monitor",
            "### Compute routing state machine",
            "#### Pre-flight gate",
            "#### Runtime monitor",
            "#### Terminal states",
        ];
        for section in &required {
            assert!(
                AGENT_MD.contains(section),
                "metal-agent.md missing section: {section}"
            );
        }
    }

    #[test]
    fn hints_md_has_required_sections() {
        let required = [
            "## Kernel Catalog",
            "## Operation Routing",
            "## Fusion Patterns",
            "## Implementation Hints",
            "## Pipeline Recipes",
            "## Performance Estimation Reference",
        ];
        for section in &required {
            assert!(
                HINTS_MD.contains(section),
                "metal-kernel-hints.md missing section: {section}"
            );
        }
    }

    #[test]
    fn hints_md_has_all_kernel_domains() {
        let domains = [
            "### Reductions",
            "### Correlation & Covariance",
            "### Elementwise Unary",
            "### ML Activations",
            "### Elementwise Binary",
            "### Scans",
            "### Compaction",
            "### Dot Products & Similarity",
            "### Quantization",
            "### Layout Transforms",
            "### Fused Map-Reduce",
            "### Advanced Statistics & Sorting",
            "### Softmax",
            "### Normalization",
            "### Attention & Positional Encoding",
            "### Loss Functions",
            "### Optimizers & Training",
            "### FFT & Spectral",
            "### Signal Processing",
            "### Linear Algebra",
            "### Simulation & Physics",
            "### Random Number Generation",
        ];
        for domain in &domains {
            assert!(
                HINTS_MD.contains(domain),
                "metal-kernel-hints.md missing kernel domain: {domain}"
            );
        }
    }

    #[test]
    fn agent_md_has_all_terminal_states() {
        let states = ["T0: CPU_INLINE", "T1: LOCAL_METAL", "T2: LOCAL_CPU_RUST",
                       "T3: LAMBDA_H100", "T4: AWS_MULTI_NODE", "T5: DO_SINGLE", "T6: DO_FLEET"];
        for state in &states {
            assert!(
                AGENT_MD.contains(state),
                "metal-agent.md missing terminal state: {state}"
            );
        }
    }

    #[test]
    fn agent_md_has_health_monitor_probes() {
        let probes = ["memory_pressure", "pmset -g therm", "vm.loadavg", "CPU_Speed_Limit"];
        for probe in &probes {
            assert!(
                AGENT_MD.contains(probe),
                "metal-agent.md missing health probe: {probe}"
            );
        }
    }

    // ── 2. CLI behavior: init ─────────────────────────────────────────

    #[test]
    fn init_creates_agent_files() {
        let dir = test_dir("init-creates");
        // Add .git so we don't get the warning
        fs::create_dir_all(dir.join(".git")).unwrap();

        init(&dir, false);

        let agent = dir.join(".claude/agents/metal.md");
        let hints = dir.join(".claude/agents/metal-kernel-hints.md");
        assert!(agent.exists(), "metal.md should be created");
        assert!(hints.exists(), "metal-kernel-hints.md should be created");

        let agent_content = fs::read_to_string(&agent).unwrap();
        let hints_content = fs::read_to_string(&hints).unwrap();
        assert_eq!(agent_content, AGENT_MD);
        assert_eq!(hints_content, HINTS_MD);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn init_creates_directory_structure() {
        let dir = test_dir("init-dirs");
        fs::create_dir_all(dir.join(".git")).unwrap();

        assert!(!dir.join(".claude").exists());
        init(&dir, false);
        assert!(dir.join(".claude/agents").is_dir());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_reports_missing_files() {
        let dir = test_dir("check-missing");
        fs::create_dir_all(&dir).unwrap();

        // check() prints to stdout — just verify it doesn't panic on missing files
        check(&dir);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_reports_existing_files() {
        let dir = test_dir("check-existing");
        fs::create_dir_all(dir.join(".git")).unwrap();

        init(&dir, false);
        // Should not panic, files exist
        check(&dir);

        let _ = fs::remove_dir_all(&dir);
    }

    // ── 3. Idempotency ───────────────────────────────────────────────

    #[test]
    fn init_skips_existing_without_force() {
        let dir = test_dir("skip-existing");
        fs::create_dir_all(dir.join(".git")).unwrap();

        init(&dir, false);

        // Tamper with the file to verify it's NOT overwritten
        let agent_path = dir.join(".claude/agents/metal.md");
        fs::write(&agent_path, "tampered").unwrap();

        init(&dir, false);

        let content = fs::read_to_string(&agent_path).unwrap();
        assert_eq!(content, "tampered", "init without --force should not overwrite");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn init_force_overwrites_existing() {
        let dir = test_dir("force-overwrite");
        fs::create_dir_all(dir.join(".git")).unwrap();

        init(&dir, false);

        // Tamper with the file
        let agent_path = dir.join(".claude/agents/metal.md");
        fs::write(&agent_path, "tampered").unwrap();

        init(&dir, true);

        let content = fs::read_to_string(&agent_path).unwrap();
        assert_eq!(content, AGENT_MD, "init --force should overwrite");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn init_is_idempotent_with_force() {
        let dir = test_dir("idempotent-force");
        fs::create_dir_all(dir.join(".git")).unwrap();

        init(&dir, true);
        let content_1 = fs::read_to_string(dir.join(".claude/agents/metal.md")).unwrap();

        init(&dir, true);
        let content_2 = fs::read_to_string(dir.join(".claude/agents/metal.md")).unwrap();

        assert_eq!(content_1, content_2, "repeated --force should produce identical files");

        let _ = fs::remove_dir_all(&dir);
    }

    // ── write_file unit tests ─────────────────────────────────────────

    #[test]
    fn write_file_creates_new() {
        let dir = test_dir("write-new");
        let path = dir.join("test.md");

        let wrote = write_file(&path, "hello", false);
        assert!(wrote);
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_file_skips_existing() {
        let dir = test_dir("write-skip");
        let path = dir.join("test.md");
        fs::write(&path, "original").unwrap();

        let wrote = write_file(&path, "new content", false);
        assert!(!wrote);
        assert_eq!(fs::read_to_string(&path).unwrap(), "original");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_file_force_overwrites() {
        let dir = test_dir("write-force");
        let path = dir.join("test.md");
        fs::write(&path, "original").unwrap();

        let wrote = write_file(&path, "new content", true);
        assert!(wrote);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new content");

        let _ = fs::remove_dir_all(&dir);
    }
}
