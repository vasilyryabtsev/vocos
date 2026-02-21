import argparse
import zipfile
from pathlib import Path

import gdown

from src.utils.io_utils import ROOT_PATH

PRETRAINED_GDRIVE_ID = "1PpTLsKkLIZ69lmmoHMl8n_Rhbtt6jPCp"
PRETRAINED_DIR = ROOT_PATH / "pretrained"
PRETRAINED_ZIP = ROOT_PATH / "pretrained.zip"


def _find_checkpoint(directory: Path) -> Path:
    """Return the first .pth file found recursively inside directory."""
    checkpoints = sorted(directory.rglob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoint found inside {directory}")
    return checkpoints[0]


def download_pretrained_weights(dest: str | None = None) -> str:
    """Download and unpack pretrained weights from Google Drive if needed.

    Args:
        dest: directory where weights will be saved. Defaults to PRETRAINED_DIR.
    """
    target_dir = Path(dest) if dest is not None else PRETRAINED_DIR
    zip_path = target_dir.parent / "pretrained.zip"

    if target_dir.exists():
        checkpoint = _find_checkpoint(target_dir)
        print(f"Using cached weights: {checkpoint}")
        return str(checkpoint)

    print(f"Downloading pretrained weights archive to {zip_path} ...")
    url = f"https://drive.google.com/uc?id={PRETRAINED_GDRIVE_ID}"
    gdown.download(url, str(zip_path), quiet=False)

    print(f"Extracting archive to {target_dir} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    zip_path.unlink()

    checkpoint = _find_checkpoint(target_dir)
    print(f"Weights ready: {checkpoint}")
    return str(checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
    )
    args = parser.parse_args()

    download_pretrained_weights(args.checkpoint)


if __name__ == "__main__":
    main()
