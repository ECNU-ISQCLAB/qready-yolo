from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nets.yolo import YoloBody
from utils.utils import get_anchors, get_classes


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main():
    section("1. Package versions")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"opencv: {cv2.__version__}")
    print(f"Pillow: {Image.__version__}")

    section("2. CUDA check")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch CUDA runtime: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("CUDA is not available. Check NVIDIA driver, CUDA runtime choice, and PyTorch install.")

    section("3. Project files check")
    classes_path = ROOT / "model_data" / "voc_classes.txt"
    anchors_path = ROOT / "model_data" / "yolo_anchors.txt"
    weights_path = ROOT / "model_data" / "yolov7_weights.pth"
    train_txt = ROOT / "2007_train.txt"
    val_txt = ROOT / "2007_val.txt"

    for path in (classes_path, anchors_path, weights_path, train_txt, val_txt):
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"exists: {path.relative_to(ROOT)}")

    class_names, num_classes = get_classes(str(classes_path))
    anchors, num_anchors = get_anchors(str(anchors_path))
    print(f"num_classes: {num_classes}")
    print(f"first classes: {class_names[:5]}")
    print(f"anchors shape: {anchors.shape}, num_anchors: {num_anchors}")

    section("4. Weight compatibility check")
    model = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes, "l", pretrained=False)
    model_dict = model.state_dict()
    ckpt = torch.load(weights_path, map_location="cpu")
    matched = 0
    mismatched = []
    for key, value in ckpt.items():
        key = key.replace("module.", "")
        if key in model_dict and tuple(model_dict[key].shape) == tuple(value.shape):
            matched += 1
        elif key in model_dict:
            mismatched.append((key, tuple(value.shape), tuple(model_dict[key].shape)))
    print(f"matched tensors: {matched}/{len(ckpt)}")
    print(f"shape mismatches: {len(mismatched)}")
    if mismatched[:6]:
        print("first expected mismatches, usually VOC head vs COCO head:")
        for item in mismatched[:6]:
            print(f"  {item[0]}: checkpoint {item[1]} vs model {item[2]}")
    if matched == 0:
        raise RuntimeError("No tensors matched. The weight file does not match this repo/model.")

    section("5. Forward pass check")
    device = torch.device("cuda")
    model.to(device).eval()
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        outputs = model(dummy)
    print(f"output shapes: {[tuple(output.shape) for output in outputs]}")
    print("Server environment verification passed.")


if __name__ == "__main__":
    main()
