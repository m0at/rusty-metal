# rusty-metal

CLI tool that installs a Metal GPU compute specialist agent into `.claude/agents/` for any project.

## Build & Test

```bash
cargo build --release
cargo test
```

## Architecture

This is a **content delivery tool**, not a compute project. The value is in the markdown files.

- `src/main.rs` — Zero-dependency CLI. Embeds content at compile time via `include_str!`
- `content/metal-agent.md` — Agent system prompt (architecture, dispatch conventions, performance rules, compute routing)
- `content/metal-kernel-hints.md` — 109-kernel catalog (routing tables, fusion patterns, implementation hints, pipeline recipes)

**Edit content in `content/`, not in `.claude/agents/`.** The agents directory files are outputs — they get overwritten by `rusty-metal init`.

## Key Conventions

- Zero external dependencies — keep it that way
- Content files are the product; the Rust code is just the delivery mechanism
- `--force` overwrites existing files; default is skip-if-exists
- Warn (don't error) when no `.git` or `CLAUDE.md` found in target directory
- All file writes go through the `write_file` helper for consistent output formatting
