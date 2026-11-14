"""Verify all preparation artifacts against the checked-in reproduction contract."""

from pathlib import Path

from mcptox_gateway.dataset import sha256_bytes
from mcptox_gateway.pipelines.common import OUTPUTS, stage_root, verify_inputs
from mcptox_gateway.pipelines.download_data import download_sources


def verify(root: Path) -> None:
    """Reject missing artifacts, modified settings, code drift, or changed subsets."""
    download_sources(root)
    lock = verify_inputs(root)
    if set(lock["outputs"]) != set(OUTPUTS):
        raise ValueError("Incomplete output inventory in reproduction lock")
    for relative in OUTPUTS:
        if sha256_bytes((root / relative).read_bytes()) != lock["outputs"][relative]:
            raise ValueError(f"Prepared artifact checksum mismatch: {relative}")


if __name__ == "__main__":
    verify(stage_root(__doc__))
    print("Preparation verified. Model qualification and independent review remain required.")
