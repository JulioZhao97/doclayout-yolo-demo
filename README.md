# DocLayout-YOLO 文档版面分析 Demo

Gradio demo：上传文档页面图像，输出 22 类版面元素检测框（按类别着色，含置信度与 NMS 阈值滑杆）。

权重：MinerU stage2 微调 ft-v1 best（val mAP50-95 0.58345）。

## 用法

- 推理入口：`app.py`（Gradio）
- 权重文件：`best.pt` 需放在仓库根目录（119MB，超 GitHub 单文件限制，请通过平台网页上传或 LFS 传输）
- `ultralytics/` 为本项目自定义的 ultralytics 代码树（含 HWAttn 模块），app.py 通过 sys.path 优先加载它
- 依赖见 `requirements.txt`（torch 由运行环境预装，无需重复安装）
