import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
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
    """Label chip: white core, class-colored border and text, placed above the box top-left."""
    fs = max(0.42, 0.0005 * max(img.shape[:2]))
    ft = max(1, th)
    (tw, th_), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    pad = max(2, th + 1)
    chip_h = th_ + _bl + 2 * pad
    chip_w = tw + 2 * pad
    cy1 = y - chip_h - 1
    if cy1 < 0:
        cy1 = min(y + 1, img.shape[0] - chip_h)
    cx1 = max(0, min(x, img.shape[1] - chip_w - 1))
    cv2.rectangle(img, (cx1, cy1), (cx1 + chip_w, cy1 + chip_h), color, -1)
    cv2.rectangle(img, (cx1 + th, cy1 + th), (cx1 + chip_w - th, cy1 + chip_h - th), (255, 255, 255), -1)
    cv2.rectangle(img, (cx1, cy1), (cx1 + chip_w, cy1 + chip_h), color, th)
    cv2.putText(img, text, (cx1 + pad, cy1 + chip_h - pad - _bl),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft, cv2.LINE_AA)


def detect(img, conf, nms_iou):
    if img is None:
        return None, ""
    img = img[:, :, ::-1].copy()  # RGB -> BGR
    res = MODEL.predict(img, imgsz=1280, conf=float(conf), iou=float(nms_iou),
                        verbose=False)[0]
    boxes = res.boxes
    lw = max(1, round(0.0012 * max(img.shape[:2])))
    out = img.copy()
    lines = []
    order = np.argsort(-boxes.conf.cpu().numpy()) if len(boxes) else []
    for i in order:
        c = int(boxes.cls[i])
        conf_v = float(boxes.conf[i])
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        color = PALETTE[c % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)
        _chip(out, f"{CLASS_NAMES[c]} {conf_v:.2f}", x1, y1, color, lw)
        lines.append(f"{CLASS_NAMES[c]}  {conf_v:.2f}")
    summary = f"检测到 {len(lines)} 个目标" + ("：\n" + "\n".join(lines) if lines else "")
    return out[:, :, ::-1], summary


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="上传文档图像"),
        gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度阈值 conf"),
        gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="NMS IoU 阈值"),
    ],
    outputs=[
        gr.Image(type="numpy", label="检测结果"),
        gr.Textbox(label="目标列表", lines=8),
    ],
    title="DocLayout-YOLO 文档版面分析 Demo",
    description="上传文档页面图像，模型输出 22 类版面元素的检测框（按类别着色）。"
                "当前权重：MinerU stage2 微调 ft-v1 best（mAP50-95 0.58345）。",
    examples=[],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
