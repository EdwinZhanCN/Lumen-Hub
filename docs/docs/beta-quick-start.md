---
sidebar_position: 2
---

# Beta Quick Start

Install the CLI, initialize a config, then let `lumen-cli start` download the matching Lumen Hub runtime from the release manifest.

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

The installer only installs `lumen-cli`. It does not download the hub runtime or write a Lumen config.
