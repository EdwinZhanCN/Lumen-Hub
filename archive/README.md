# Archive

This directory keeps code that is no longer part of the active Lumen workspace.

## lumen-shell

`archive/lumen-shell` contains the archived desktop shell crate and its packaging
configs. It was removed from the active build graph because distribution for
desktop same-machine scenarios is now handled by the Lumilio desktop client,
while server deployments are handled through Docker and the CLI.

The code is retained so it can be revived if the distribution decision changes.
To reactivate it, move the crate back under `crates/`, restore the packaging
configs under `packaging/` or update their paths, then restore the CI and release
jobs that build and publish the desktop shell assets.
