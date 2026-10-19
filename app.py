import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")


def _resolve_lfs_pointer(path):
    """Git checkouts without git-lfs leave a pointer text file; pull the real weights."""
    with open(path, "rb") as f:
        head = f.read(64)
    if not head.startswith(b"version https://git-lfs"):
        return
    print("[demo] best.pt is a Git LFS pointer, pulling real weights ...", flush=True)
    import subprocess
    root = os.path.dirname(path)
    r = subprocess.run(["git", "lfs", "pull", "--include", "best.pt"], cwd=root,
                       capture_output=True, text=True)
    ok = not os.path.getsize(path) < 1000 and not open(path, "rb").read(20).startswith(b"version")
    if r.returncode != 0 or not ok:
        raise RuntimeError(
            "best.pt is a Git LFS pointer and 'git lfs pull' failed:\n"
            + (r.stderr or r.stdout or "git-lfs not available")
            + "\nFix: upload best.pt via the Space web file manager (not git), "
              "or push with git-lfs installed."
        )
    print("[demo] best.pt restored from LFS.", flush=True)


_resolve_lfs_pointer(MODEL_PATH)
MODEL = YOLO(MODEL_PATH)

CLASS_NAMES = [
    "text", "title", "header", "footer", "page_number", "image",
    "image_caption", "image_footnote", "image_block", "table", "table_caption",
    "table_footnote", "equation", "equation_block", "list", "ref_text",
    "aside_text", "page_footnote", "code", "code_caption", "algorithm", "phonetic",
]

PALETTE = [
    (66, 170, 255), (255, 170, 66), (66, 255, 170), (170, 66, 255), (255, 66, 170),
    (66, 220, 220), (220, 220, 66), (220, 66, 220), (66, 66, 255), (255, 128, 0),
    (0, 255, 128), (128, 0, 255), (255, 255, 0), (0, 128, 255), (128, 255, 0),
    (0, 255, 255), (255, 0, 255), (192, 192, 192), (0, 0, 128), (128, 0, 0),
    (0, 128, 0), (128, 128, 0),
]


def _chip(img, text, x, y, color, th):
    """detectron2-style label: solid class-color bar with white text, above the box top-left."""
    fs = max(0.42, 0.0005 * max(img.shape[:2]))
    ft = 1
    (tw, th_), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    pad = max(2, th)
    chip_h = th_ + _bl + 2 * pad
    chip_w = tw + 2 * pad
    cy1 = y - chip_h - 1
    if cy1 < 0:
        cy1 = min(y + 1, img.shape[0] - chip_h)
    cx1 = max(0, min(x, img.shape[1] - chip_w - 1))
    cv2.rectangle(img, (cx1, cy1), (cx1 + chip_w, cy1 + chip_h), color, -1)
    # white text on dark chips, black text on light ones
    luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    tcolor = (0, 0, 0) if luminance > 150 else (255, 255, 255)
    cv2.putText(img, text, (cx1 + pad, cy1 + chip_h - pad - _bl),
                cv2.FONT_HERSHEY_SIMPLEX, fs, tcolor, ft, cv2.LINE_AA)


def detect(img, conf, nms_iou):
    if img is None:
        return None
    img = img[:, :, ::-1].copy()  # RGB -> BGR
    res = MODEL.predict(img, imgsz=1024, conf=float(conf), iou=float(nms_iou),
                        verbose=False)[0]
    boxes = res.boxes
    lw = max(1, round(0.0012 * max(img.shape[:2])))
    out = img.copy()
    order = np.argsort(-boxes.conf.cpu().numpy()) if len(boxes) else []
    # detectron2-style: translucent color fill (drawn on an overlay, alpha ~0.35)
    overlay = out.copy()
    for i in order:
        c = int(boxes.cls[i])
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), PALETTE[c % len(PALETTE)], -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, dst=out)
    # thin border + label chip on top of the fill
    for i in order:
        c = int(boxes.cls[i])
        conf_v = float(boxes.conf[i])
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        color = PALETTE[c % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)
        _chip(out, f"{CLASS_NAMES[c]} {conf_v:.2f}", x1, y1, color, lw)
    return out[:, :, ::-1]


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="上传文档图像"),
        gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度阈值 conf"),
        gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="NMS IoU 阈值"),
    ],
    outputs=[
        gr.Image(type="numpy", label="检测结果"),
    ],
    title="DocLayout-YOLO: Axis-Attentive Document Layout Detector with Mesh-Guided Synthetic Pretraining",
    examples=[],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
