from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
YOLO_ROOT = REPO_ROOT / "yolov7-pytorch-master"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from nets.yolo import YoloBody  # noqa: E402
from utils.utils import cvtColor, get_anchors, get_classes, preprocess_input, resize_image  # noqa: E402
from utils.utils_bbox import DecodeBox  # noqa: E402


MODEL_PATH = YOLO_ROOT / "model_data" / "yolov7_weights.pth"
COCO_CLASSES_PATH = YOLO_ROOT / "model_data" / "coco_classes.txt"
ANCHORS_PATH = YOLO_ROOT / "model_data" / "yolo_anchors.txt"
VAL_ANN_PATH = YOLO_ROOT / "2007_val.txt"
OUTPUT_DIR = YOLO_ROOT / "codex_quantum_outputs"

PHI = "l"
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
INFER_INPUT_SHAPE = [640, 640]
CONFIDENCE = 0.35
NMS_IOU = 0.3
QUANTUM_TARGETS = ("p3", "p4", "p5")
QUANTUM_DEPTH = 2
DEFAULT_LIMIT = 30
SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lines(path: Path, limit: int) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = [_normalize_annotation_line(line) for line in lines]
    return normalized[:limit]


def _normalize_annotation_line(line: str) -> str:
    parts = line.split()
    if not parts:
        return line
    image_name = Path(parts[0]).name
    image_path = YOLO_ROOT / "VOCdevkit" / "VOC2007" / "JPEGImages" / image_name
    return " ".join([str(image_path)] + parts[1:])


def _exact_load(model: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=device)
    matched = {}
    for key, value in pretrained_dict.items():
        if key in model_dict and tuple(model_dict[key].shape) == tuple(value.shape):
            matched[key] = value
    model_dict.update(matched)
    model.load_state_dict(model_dict)


def build_model(
    device: torch.device,
    use_quantum: bool,
    quantum_targets: tuple[str, ...] = QUANTUM_TARGETS,
    quantum_depth: int = QUANTUM_DEPTH,
) -> YoloBody:
    coco_classes, _ = get_classes(str(COCO_CLASSES_PATH))
    model = YoloBody(
        ANCHORS_MASK,
        len(coco_classes),
        PHI,
        pretrained=False,
        use_quantum=use_quantum,
        quantum_targets=quantum_targets,
        quantum_depth=quantum_depth,
    )
    _exact_load(model, MODEL_PATH, device)
    model.to(device)
    return model.fuse().eval()


def preprocess_pil_image(image: Image.Image, input_shape: list[int]) -> tuple[np.ndarray, torch.Tensor]:
    image = cvtColor(image)
    resized = resize_image(image, (input_shape[1], input_shape[0]), True)
    image_array = np.expand_dims(
        np.transpose(preprocess_input(np.array(resized, dtype="float32")), (2, 0, 1)),
        0,
    )
    return np.array(image), torch.from_numpy(image_array).float()


def run_detector(
    model: YoloBody,
    image_path: Path,
    class_names: list[str],
    bbox_util: DecodeBox,
    device: torch.device,
) -> dict[str, object]:
    image = Image.open(image_path)
    image_shape = np.array(np.shape(image)[0:2])
    _, image_tensor = preprocess_pil_image(image, INFER_INPUT_SHAPE)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(image_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        decoded = bbox_util.decode_box(outputs)
        results = bbox_util.non_max_suppression(
            torch.cat(decoded, 1),
            len(class_names),
            INFER_INPUT_SHAPE,
            image_shape,
            True,
            conf_thres=CONFIDENCE,
            nms_thres=NMS_IOU,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

    detections = []
    if results[0] is not None:
        for row in results[0]:
            left, top, right, bottom = map(float, row[:4])
            score = float(row[4] * row[5])
            class_id = int(row[6])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "score": score,
                    "box": [left, top, right, bottom],
                }
            )
        detections.sort(key=lambda item: item["score"], reverse=True)

    return {
        "image_path": str(image_path),
        "detections": detections,
        "forward_ms": (t1 - t0) * 1000.0,
        "postprocess_ms": (t2 - t1) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
    }


def box_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_shift(box1: list[float], box2: list[float]) -> float:
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    return math.hypot(c1x - c2x, c1y - c2y)


def match_same_class(
    baseline: list[dict[str, object]],
    quantum: list[dict[str, object]],
    iou_threshold: float,
) -> tuple[list[tuple[dict[str, object], dict[str, object], float]], list[dict[str, object]], list[dict[str, object]]]:
    pairs = []
    for bi, bdet in enumerate(baseline):
        for qi, qdet in enumerate(quantum):
            iou = box_iou(bdet["box"], qdet["box"])
            if iou >= iou_threshold:
                pairs.append((iou, bi, qi))
    pairs.sort(reverse=True)

    used_b = set()
    used_q = set()
    matched = []
    for iou, bi, qi in pairs:
        if bi in used_b or qi in used_q:
            continue
        used_b.add(bi)
        used_q.add(qi)
        matched.append((baseline[bi], quantum[qi], iou))

    missing = [baseline[i] for i in range(len(baseline)) if i not in used_b]
    extra = [quantum[i] for i in range(len(quantum)) if i not in used_q]
    return matched, missing, extra


def describe_detection(det: dict[str, object]) -> str:
    box = det["box"]
    return (
        f"{det['class_name']}(score={det['score']:.4f},"
        f" box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}])"
    )


def compare_detections(
    baseline: list[dict[str, object]],
    quantum: list[dict[str, object]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0001,
    shift_threshold: float = 0.05,
) -> list[str]:
    grouped_baseline: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_quantum: dict[str, list[dict[str, object]]] = defaultdict(list)
    for det in baseline:
        grouped_baseline[str(det["class_name"])].append(det)
    for det in quantum:
        grouped_quantum[str(det["class_name"])].append(det)

    messages: list[str] = []
    class_names = sorted(set(grouped_baseline) | set(grouped_quantum))
    for class_name in class_names:
        matched, missing, extra = match_same_class(
            grouped_baseline.get(class_name, []),
            grouped_quantum.get(class_name, []),
            iou_threshold=iou_threshold,
        )
        for det in missing:
            messages.append(f"{describe_detection(det)} 缺失")
        for det in extra:
            messages.append(f"{describe_detection(det)} 多余")
        for bdet, qdet, iou in matched:
            delta = float(qdet["score"]) - float(bdet["score"])
            if abs(delta) >= score_threshold:
                messages.append(
                    f"{class_name} 置信度变化({delta:+.4f})"
                )
            shift = center_shift(bdet["box"], qdet["box"])
            if shift >= shift_threshold:
                messages.append(
                    f"{class_name} 偏移(IoU={iou:.3f}, 中心偏移={shift:.2f}px)"
                )
    if not messages:
        messages.append("无明显差异")
    return messages


def build_summary_lines(
    image_ids: list[str],
    baseline_records: list[dict[str, object]],
    quantum_records: list[dict[str, object]],
) -> list[str]:
    lines: list[str] = []
    for image_id, baseline, quantum in zip(image_ids, baseline_records, quantum_records):
        diff_messages = compare_detections(
            baseline["detections"],
            quantum["detections"],
        )
        lines.append(f"pngid:{image_id}")
        lines.append(f"quantum的差异：{'；'.join(diff_messages)}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline YOLOv7 vs random quantum YOLOv7.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of images to compare.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "compare_quantumandnoquantum_summary.txt",
        help="Output summary text path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=OUTPUT_DIR / "compare_quantumandnoquantum_summary.json",
        help="Output summary json path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names, num_classes = get_classes(str(COCO_CLASSES_PATH))
    anchors, _ = get_anchors(str(ANCHORS_PATH))
    bbox_util = DecodeBox(anchors, num_classes, (INFER_INPUT_SHAPE[0], INFER_INPUT_SHAPE[1]), ANCHORS_MASK)

    baseline_model = build_model(device, use_quantum=False)
    quantum_model = build_model(
        device,
        use_quantum=True,
        quantum_targets=QUANTUM_TARGETS,
        quantum_depth=QUANTUM_DEPTH,
    )

    lines = load_lines(VAL_ANN_PATH, args.limit)
    image_paths = [Path(line.split()[0]) for line in lines]
    image_ids = [path.stem for path in image_paths]

    baseline_records = []
    quantum_records = []
    for image_path in image_paths:
        baseline_records.append(run_detector(baseline_model, image_path, class_names, bbox_util, device))
        quantum_records.append(run_detector(quantum_model, image_path, class_names, bbox_util, device))

    summary_lines = build_summary_lines(image_ids, baseline_records, quantum_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(summary_lines) + "\n", encoding="utf-8-sig")

    json_payload = []
    for image_id, baseline, quantum in zip(image_ids, baseline_records, quantum_records):
        json_payload.append(
            {
                "pngid": image_id,
                "quantum的差异": compare_detections(baseline["detections"], quantum["detections"]),
                "baseline_total_ms": baseline["total_ms"],
                "quantum_total_ms": quantum["total_ms"],
            }
        )
    args.json_output.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"对比摘要已写入: {args.output}")
    print(f"对比JSON已写入: {args.json_output}")


if __name__ == "__main__":
    main()
