# discopt Claude Code Skills

Custom slash commands for working with discopt in [Claude Code](https://claude.ai/code).

## Available commands

| Command | Description |
|---------|-------------|
| `/benchmark-report` | Analyze benchmark JSON results into a narrative report |
| `/convert` | Translate models between solver formats |
| `/diagnose` | Diagnose solver failures, infeasibility, and convergence issues |
| `/discoptbot` | Literature scanner for discopt-related papers |
| `/explain-model` | Generate mathematical documentation for a model |
| `/formulate` | Build a discopt model from a natural language description |
| `/reformulate` | Suggest model improvements (big-M, weak bounds, symmetry, etc.) |

## Install

```bash
bash claude-skills/install.sh
```

This symlinks the skill files into `.claude/commands/` so Claude Code registers them as slash commands. The `.claude/` directory is gitignored.
