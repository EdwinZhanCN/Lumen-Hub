---
sidebar_position: 2
---

# Beta Quick Start

Install `lumen-cli`, generate a hub config, then start the matching Lumen Hub runtime from the GitHub release manifest.

The installer script only places `lumen-cli` on your PATH. It does **not** download the hub binary or write a config.

## macOS / Linux

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://lumilio.org/lumen/install.sh | sh
lumen-cli init
lumen-cli start
```

## Windows

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://lumilio.org/lumen/install.ps1 | iex"
lumen-cli init
lumen-cli start
```

## What `init` creates

| Path | Purpose |
|---|---|
| `~/.lumen/config.yaml` | Hub services, models, server, batching |
| `~/.lumen/bootstrap.json` | Dist profile, backend package, manifest URL for `start` |

`init` detects OS/arch/RAM, asks for download region (`other` → huggingface.co, `cn` → hf-mirror.com), preset (model bundle), and backend (ONNX CPU, CUDA, DirectML, etc.).

## What `start` does

1. Reads `bootstrap.json` and downloads the hub archive for your platform profile if missing
2. Ensures ONNX Runtime (or Jetson GPU wheel on `linux-arm64-jetson`)
3. Launches `lumen-hub --config ~/.lumen/config.yaml`
4. On first run, downloads model artifacts into `metadata.cache_dir`

Override the manifest URL:

```bash
lumen-cli start --manifest-url https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download/manifest.json
```

## Local development

Build and run from source instead of release archives:

```bash
cargo build -p lumen-hub --features profile-universal-cpu
cargo run -p lumen-hub --features profile-universal-cpu -- \
  --config path/to/config.yaml
```

See [Beta Dist](./development/beta-local-dist) for dist profiles and [Configuration](./configuration/lumen-config) for the YAML schema.
