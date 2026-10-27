"""Run one image through one of the repository's historical checkpoints."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"detect": ROOT / "models" / "best.pt", "segment": ROOT / "SegModels" / "best.pt"}


def main():
    parser = argparse.ArgumentParser(description="Run a historical dental-image checkpoint on one local image.")
    parser.add_argument("--image", type=Path, required=True, help="Existing input image")
    parser.add_argument("--output", type=Path, required=True, help="New .png or .jpg image")
    parser.add_argument("--model", choices=WEIGHTS, default="segment")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error("input image does not exist")
    if args.image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        parser.error("input image extension must be a supported image format")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.output.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        parser.error("output extension must be .png, .jpg, or .jpeg")
    if not args.output.parent.is_dir():
        parser.error("output directory does not exist")
    if not WEIGHTS[args.model].is_file():
        parser.error("checkpoint is missing")

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS[args.model]))
    result = model.predict(source=str(args.image), imgsz=640, device="cpu", verbose=False)[0]
    ok, encoded = cv2.imencode(args.output.suffix.lower(), result.plot())
    if not ok:
        parser.error("could not encode output image")
    try:
        with args.output.open("xb") as output:
            output.write(encoded.tobytes())
    except FileExistsError:
        parser.error("output already exists; choose a new path")
    print(f"Saved {args.model} visualization to {args.output}")


if __name__ == "__main__":
    main()
