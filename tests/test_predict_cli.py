import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_missing_image_fails_before_loading_model(tmp_path):
    output = tmp_path / "prediction.png"
    result = run("--image", tmp_path / "missing.png", "--output", output)
    assert result.returncode != 0
    assert "image" in result.stderr.lower()
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"placeholder")
    output = tmp_path / "output.png"
    output.write_bytes(b"existing")
    result = run("--image", image, "--output", output)
    assert result.returncode != 0
    assert "already exists" in result.stderr.lower()
    assert output.read_bytes() == b"existing"


def test_unsupported_output_extension_fails(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"placeholder")
    result = run("--image", image, "--output", tmp_path / "output.txt")
    assert result.returncode != 0
    assert "extension" in result.stderr.lower()


@pytest.mark.parametrize("extension", [".streams", ".txt", ".mp4"])
def test_non_image_sources_are_rejected_before_loading_model(tmp_path, extension):
    source = tmp_path / f"input{extension}"
    source.write_text("not an image", encoding="utf-8")
    result = run("--image", source, "--output", tmp_path / "output.png")
    assert result.returncode != 0
    assert "input image extension" in result.stderr.lower()
