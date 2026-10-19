---
title: DocLayout-YOLO Demo
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: apache-2.0
---

DocLayout-YOLO 文档版面分析 Demo：上传文档图像，输出 22 类版面元素检测框。
当前权重：MinerU stage2 微调 1024 版 best（val mAP50-95 0.60999），imgsz 1024。

## 开发者说明

- 推理入口：`app.py`（Gradio）
- 权重文件：`best.pt` 放在仓库根目录（119MB，经 git-lfs 传输；app.py 内置 LFS pointer 自愈逻辑）
- `ultralytics/` 为本项目自定义的 ultralytics 代码树（含 HWAttn 模块），app.py 通过 sys.path 优先加载它
- 依赖见 `requirements.txt`（torch 由运行环境预装，无需重复安装）
