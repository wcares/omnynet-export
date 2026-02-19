# Asset Signing

Ed25519 signing for OmnyNet fortress integrity verification.

All assets downloaded by the agent (models, processors, binaries) are verified against `.sig` companion files before use. This module produces those signatures.

## Quick Start

```bash
# Sign a model after export
omnynet-export sign model.omny --key ~/.omnynet-keys/omnynet-signing.key

# Sign multiple files at once
omnynet-export sign *.omny processors/* --key ~/.omnynet-keys/omnynet-signing.key

# Verify before uploading to CDN
omnynet-export verify model.omny --pubkey ~/.omnynet-keys/omnynet-signing.pub
```

## Key Management

### Generate a new keypair

```bash
omnynet-export generate-key --out ~/.omnynet-keys/
```

Produces:
- `omnynet-signing.key` — 32-byte raw Ed25519 private key (mode 0600)
- `omnynet-signing.pub` — hex-encoded public key

### Embed public key in fortress

```bash
omnynet-export show-pubkey --key ~/.omnynet-keys/omnynet-signing.key
```

Outputs a Rust `[u8; 32]` literal to paste into `fortress.rs`:

```rust
const OMNYNET_PUBLIC_KEY: [u8; 32] = [
    0x1a, 0x2b, 0x3c, ...
];
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `sign <files> --key <path>` | Sign files, producing `.sig` companions |
| `verify <files> --pubkey <path>` | Verify files against their `.sig` companions |
| `generate-key --out <dir>` | Generate new Ed25519 keypair |
| `show-pubkey --key <path>` | Print public key as Rust literal |

## .sig File Format

74 bytes, binary:

```
Offset  Size  Field
0       4     Magic: "OSIG" (0x4f534947)
4       2     Version: 1 (u16 LE)
6       4     Flags: 0 (reserved)
10      64    Ed25519 signature
```

The signature is over the **SHA256 hash** of the file contents (not the raw file).

This format is identical to what `fortress.rs` expects. Signatures produced by the Python exporter and the Rust `omnynet-sign` binary are fully interchangeable.

## How Verification Works (Agent Side)

When the agent downloads any asset via `DownloadService`:

1. Download the file to a temp path
2. Rename temp to final path
3. Fetch `<original_url>.sig` from CDN
4. Parse the `.sig` file (validate magic, version)
5. SHA256 the downloaded file
6. Ed25519 verify the hash against the signature using the hardcoded public key
7. If valid: store local integrity stamp (`.sha256` file)
8. If invalid: **delete the downloaded file**

Code path: `DownloadService::download()` → `fortress::verify_and_store()`

## CDN Layout

All assets on CDN must have a `.sig` companion at `<asset_path>.sig`:

```
omnytron/omnynet-models/
├── processors/
│   ├── clip-processor
│   ├── clip-processor.sig
│   ├── semantic-processor
│   ├── semantic-processor.sig
│   └── ...
├── semantic/
│   ├── semantic-v1.omny
│   └── semantic-v1.omny.sig
├── clip/
│   ├── clip-vision-v3.omny
│   └── clip-vision-v3.omny.sig
└── ...
```

## Publish Workflow

```bash
# 1. Export model
omnynet-export export-v2 model.onnx -o model.omny \
    --task embedding --processor native --processor-id semantic

# 2. Sign
omnynet-export sign model.omny --key ~/.omnynet-keys/omnynet-signing.key

# 3. Upload to CDN (both file + .sig)
mc cp model.omny omnytron/omnynet-models/semantic/model.omny
mc cp model.omny.sig omnytron/omnynet-models/semantic/model.omny.sig
```

## Python API

```python
from omnynet_export.signing import (
    load_signing_key,
    sign_and_write,
    verify_file,
    generate_keypair,
    sig_path_for,
)
from pathlib import Path

# Sign
key = load_signing_key(Path("~/.omnynet-keys/omnynet-signing.key"))
sig_path = sign_and_write(Path("model.omny"), key)

# Verify
from omnynet_export.signing import load_public_key
pubkey = load_public_key(Path("~/.omnynet-keys/omnynet-signing.pub"))
verify_file(Path("model.omny"), sig_path, pubkey)  # raises on failure
```

## Security Notes

- The private key must NEVER be committed to git or shipped in binaries
- The public key IS embedded in the agent binary (`fortress.rs`) — this is intentional
- Signing happens at build/CI time only. The agent only verifies, never signs
- If a key is compromised, rotate: generate new keypair, re-sign all CDN assets, rebuild agent with new public key
