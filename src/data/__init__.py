"""Shared EPL dataset location and portable content fingerprint."""
import hashlib
from pathlib import Path

MATCHES_PATH = Path("data/processed/epl_matches.csv")


def dataset_sha256(path=MATCHES_PATH):
    """Hash CSV content independently of Windows/Unix line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
