"""
Ed25519 asset signing for OmnyNet fortress integrity.

Signs assets with Ed25519 (SHA256 hash → Ed25519 signature).
Produces .sig companion files compatible with the agent's fortress verifier.

Sig file format (74 bytes):
    [0:4]   Magic: b"OSIG"
    [4:6]   Version: u16 LE (1)
    [6:10]  Flags: u32 LE (reserved, 0)
    [10:74] Signature: 64 bytes Ed25519
"""

import hashlib
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization


# Must match fortress.rs
SIG_MAGIC = b"OSIG"
SIG_VERSION = 1
SIG_FILE_SIZE = 74


def load_signing_key(path: Path) -> Ed25519PrivateKey:
    """Load a raw 32-byte Ed25519 signing key."""
    key_bytes = path.read_bytes()
    if len(key_bytes) != 32:
        raise ValueError(f"Invalid key file: expected 32 bytes, got {len(key_bytes)}")
    return Ed25519PrivateKey.from_private_bytes(key_bytes)


def load_public_key(path: Path) -> Ed25519PublicKey:
    """Load a hex-encoded Ed25519 public key."""
    hex_str = path.read_text().strip()
    key_bytes = bytes.fromhex(hex_str)
    if len(key_bytes) != 32:
        raise ValueError(f"Invalid public key: expected 32 bytes, got {len(key_bytes)}")
    return Ed25519PublicKey.from_public_bytes(key_bytes)


def get_public_key(private_key: Ed25519PrivateKey) -> Ed25519PublicKey:
    """Derive public key from private key."""
    return private_key.public_key()


def sha256_file(path: Path) -> bytes:
    """SHA256 hash of a file (streaming, memory-efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(256 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()


def sign_file(path: Path, private_key: Ed25519PrivateKey) -> bytes:
    """Sign a file, returning the 74-byte .sig content.

    The signature is over the SHA256 hash of the file (matching fortress.rs).
    """
    file_hash = sha256_file(path)
    signature = private_key.sign(file_hash)

    # Build sig file: magic + version + flags + signature
    sig_data = bytearray(SIG_FILE_SIZE)
    sig_data[0:4] = SIG_MAGIC
    sig_data[4:6] = SIG_VERSION.to_bytes(2, "little")
    sig_data[6:10] = (0).to_bytes(4, "little")  # flags reserved
    sig_data[10:74] = signature

    return bytes(sig_data)


def verify_file(path: Path, sig_path: Path, public_key: Ed25519PublicKey) -> bool:
    """Verify a file against its .sig companion.

    Returns True if valid, raises on invalid.
    """
    sig_data = sig_path.read_bytes()
    if len(sig_data) != SIG_FILE_SIZE:
        raise ValueError(f"Bad sig file size: {len(sig_data)} != {SIG_FILE_SIZE}")

    if sig_data[0:4] != SIG_MAGIC:
        raise ValueError("Bad sig magic")

    version = int.from_bytes(sig_data[4:6], "little")
    if version != SIG_VERSION:
        raise ValueError(f"Unsupported sig version: {version}")

    signature = sig_data[10:74]
    file_hash = sha256_file(path)

    # Ed25519 verify raises InvalidSignature on failure
    public_key.verify(signature, file_hash)
    return True


def sig_path_for(file_path: Path) -> Path:
    """Get the .sig companion path for a file."""
    return file_path.parent / (file_path.name + ".sig")


def sign_and_write(file_path: Path, private_key: Ed25519PrivateKey) -> Path:
    """Sign a file and write the .sig companion. Returns sig path."""
    sig_data = sign_file(file_path, private_key)
    out_path = sig_path_for(file_path)
    out_path.write_bytes(sig_data)
    return out_path


def generate_keypair(output_dir: Path) -> tuple[Path, Path]:
    """Generate a new Ed25519 keypair.

    Returns (private_key_path, public_key_path).
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw 32-byte private key
    key_path = output_dir / "omnynet-signing.key"
    raw_private = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path.write_bytes(raw_private)
    key_path.chmod(0o600)

    # Save public key as hex
    pub_path = output_dir / "omnynet-signing.pub"
    raw_public = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pub_path.write_text(raw_public.hex())

    return key_path, pub_path


def public_key_as_rust(private_key: Ed25519PrivateKey) -> str:
    """Format public key as Rust [u8; 32] array literal."""
    public_key = private_key.public_key()
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    lines = ["const OMNYNET_PUBLIC_KEY: [u8; 32] = ["]
    for i in range(0, 32, 8):
        chunk = raw[i : i + 8]
        hex_strs = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hex_strs},")
    lines.append("];")
    return "\n".join(lines)
