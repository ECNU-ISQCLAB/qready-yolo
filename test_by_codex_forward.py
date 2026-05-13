from pathlib import Path
import contextlib
import io
import re

import torch
from PIL import Image, ImageDraw

from yolo import YOLO


ROOT = Path(__file__).resolve().parent
VOC_ROOT = ROOT / "VOCdevkit" / "VOC2007"
OUTPUT_DIR = ROOT / "codex_forward_outputs"
IMAGE_LIMIT = 100
CONFIDENCE = 0.35
SPLIT_NAME = "val"


def patch_pillow_textsize():
    """Keep this old repo working on Pillow 10+, where ImageDraw.textsize was removed."""
    if hasattr(ImageDraw.ImageDraw, "textsize"):
        return

    def textsize(self, text, font=None, *args, **kwargs):
        left, top, right, bottom = self.textbbox((0, 0), text, font=font, *args, **kwargs)
        return right - left, bottom - top

    ImageDraw.ImageDraw.textsize = textsize


def patch_torch_from_numpy():
    """Work around a local torch/numpy bridge issue without editing project code."""
    original_from_numpy = torch.from_numpy

    def safe_from_numpy(array):
        try:
            return original_from_numpy(array)
        except TypeError as exc:
            if "expected np.ndarray" not in str(exc):
                raise
            dtype = torch.float32 if str(getattr(array, "dtype", "")) == "float32" else None
            if dtype is None:
                return torch.tensor(array.tolist())
            return torch.tensor(array.tolist(), dtype=dtype)

    torch.from_numpy = safe_from_numpy


def read_ids(split_name="val", limit=5):
    split_path = VOC_ROOT / "ImageSets" / "Main" / f"{split_name}.txt"
    ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:limit]


def image_path_for_id(image_id):
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        path = VOC_ROOT / "JPEGImages" / f"{image_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for VOC id {image_id}")


def main():
    patch_pillow_textsize()
    patch_torch_from_numpy()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 88)
    print("Forward-only pretrained YOLOv7 check")
    print("=" * 88)
    print("This script does NOT train.")
    print("It loads model_data/yolov7_weights.pth with COCO classes and runs inference on VOC images.")
    print("Because the model is still COCO-pretrained, this is a baseline visual check before VOC fine-tuning.")
    print(f"split: {SPLIT_NAME}")
    print(f"image_limit: {IMAGE_LIMIT}")
    print(f"confidence: {CONFIDENCE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_path = ROOT / "model_data" / "yolov7_weights.pth"
    classes_path = ROOT / "model_data" / "coco_classes.txt"
    print(f"model_path: {model_path}")
    print(f"classes_path: {classes_path}")
    print(f"output_dir: {OUTPUT_DIR}")

    yolo = YOLO(
        model_path=str(model_path),
        classes_path=str(classes_path),
        cuda=torch.cuda.is_available(),
        confidence=CONFIDENCE,
    )

    image_ids = read_ids(split_name=SPLIT_NAME, limit=IMAGE_LIMIT)
    print("\nSelected VOC image ids:", image_ids)
    print("\nDetections printed below are from yolo.detect_image().")

    summary_rows = []
    detection_pattern = re.compile(r"b'(.+?) ([0-9.]+)'\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")

    for image_id in image_ids:
        image_path = image_path_for_id(image_id)
        image = Image.open(image_path)
        print("\n" + "-" * 88)
        print(f"image id: {image_id}")
        print(f"image path: {image_path}")

        detection_log = io.StringIO()
        with contextlib.redirect_stdout(detection_log):
            result = yolo.detect_image(image)

        detection_text = detection_log.getvalue().strip()
        print(detection_text if detection_text else "no detections above threshold")

        save_path = OUTPUT_DIR / f"{image_id}_pred.png"
        result.save(save_path, quality=95, subsampling=0)
        print(f"saved result: {save_path}")

        detections = []
        for line in detection_text.splitlines():
            match = detection_pattern.search(line)
            if match:
                cls_name, score, top, left, bottom, right = match.groups()
                detections.append((cls_name, score, top, left, bottom, right))

        if detections:
            for cls_name, score, top, left, bottom, right in detections:
                summary_rows.append([image_id, cls_name, score, top, left, bottom, right, str(save_path)])
        else:
            summary_rows.append([image_id, "NO_DETECTION", "", "", "", "", "", str(save_path)])

    summary_path = OUTPUT_DIR / "forward_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("image_id\tclass\tscore\ttop\tleft\tbottom\tright\toutput_path\n")
        for row in summary_rows:
            f.write("\t".join(row) + "\n")

    print("\n" + "=" * 88)
    print("Forward check finished.")
    print(f"images processed: {len(image_ids)}")
    print(f"summary file: {summary_path}")
    print(f"Open the PNG files in: {OUTPUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()
