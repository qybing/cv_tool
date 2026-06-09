# AI 开发规范

> 本文档适用于 Cursor、Claude Code 及其他 AI 编程助手。
> **启用方式**：Cursor 粘贴到 Settings → Rules → User Rules；Claude Code 粘贴到 `CLAUDE.md` 或对话中 `@AI_DEVELOPMENT_STANDARDS.md` 引用。

---

# 一、通用开发规范

## 1.1 语言与交流

- 所有面向用户的**回复**必须使用简体中文
- 内部**思考、推理、分析、规划**过程也必须使用简体中文
- 在输出任何内容之前，先用简体中文完成内部分析；若存在思考/推理区块，该区块也必须为简体中文
- 除非用户明确要求使用其他语言，否则始终使用中文
- 代码标识符、命令、API 名称、文件路径、错误堆栈保持原文，不要翻译
- 技术术语首次出现时可附英文原文，例如：「推理过程（reasoning）」

## 1.2 开发语言

- 主要使用 **Python** 开发，需要编写代码实现时默认使用 **Python 3.10+**
- 优先使用 Python 标准库；需要第三方库时，选用常见、维护活跃的包
- 脚本、工具、数据处理、自动化任务优先写成可运行的 Python 脚本
- 遵循 PEP 8，关键函数加类型注解
- 修改现有非 Python 文件时保持原语言，不擅自改写为 Python
- 用户明确要求其他语言或技术栈时，以用户要求为准

## 1.3 代码注释

- 新增或修改代码时，注释必须使用**简体中文**
- 函数、类、复杂逻辑、非显而易见的业务规则需要加中文注释
- 注释说明意图和原因，避免复述代码本身
- 不要写无意义注释（如 `# 导入模块`、`# 定义变量`）
- 变量名、函数名、类名保持英文，不翻译

---

# 二、计算机视觉 FastAPI Web 服务规范

> 适用于将 CV 模型（检测、分割、分类等）封装为 HTTP 服务。
> 需要 Web 服务时默认使用 **FastAPI**。
> 采用**方案 A**：`app/` 放服务代码，仓库根目录放测试、脚本与工程配置。

## 2.1 核心原则

1. **够用就好**：按项目规模选单模型或多模型结构，不提前过度分层。
2. **入口与路由合并**：接口 ≤3 个时，路由写在 `main.py`；变多再拆 `routers/`。
3. **路由与推理分离**：路由只处理 HTTP，推理逻辑放 `detector.py` / `models/` / `pipeline.py`。
4. **模型只加载一次**：在 `lifespan` 启动时加载，挂到 `app.state`，禁止每请求重新加载。
5. **配置分离**：`.env` 管环境变量（端口、设备等）；`config.yaml` 管多模型与流水线（仅多模型时）。
6. **权重放 `app/weights/`**：加入 `.gitignore`，不入库。
7. **训练代码不放 app/**：训练脚本放根目录或 `scripts/`。
8. **没有 Web 需求时**：保持脚本/Notebook，不强行建 FastAPI 项目。

---

## 2.2 单模型目录结构（默认）

适用于：单个 YOLO 检测、单分割模型等。

```text
yolo-detect-service/              # 仓库根目录
├── app/                          # 服务本体
│   ├── __init__.py
│   ├── main.py                   # 入口 + 路由 + lifespan
│   ├── config.py                 # 读取 .env
│   ├── schemas.py                # 请求/响应
│   ├── detector.py               # 推理逻辑（不依赖 FastAPI）
│   ├── logger.py                 # 日志配置
│   ├── utils/                    # 工具包
│   │   ├── __init__.py
│   │   ├── image.py              # 图片处理
│   │   ├── timer.py              # 耗时统计
│   │   └── file.py               # 文件校验等
│   └── weights/                  # 模型权重
│       └── yolov8n.pt
├── tests/
│   └── test_detect.py
├── scripts/                      # 可选：export_onnx.py 等
├── train.py                      # 可选：训练脚本
├── requirements.txt
├── .env.example
├── Dockerfile                    # 可选：部署时添加
├── .gitignore
└── README.md
```

| 文件/目录 | 干什么 |
|-----------|--------|
| `main.py` | 创建 FastAPI、lifespan 加载模型、定义路由 |
| `config.py` | 从 `.env` 读取模型路径、设备、阈值等 |
| `schemas.py` | Pydantic 请求/响应模型 |
| `detector.py` | 封装模型推理，可被测试直接调用 |
| `logger.py` | `setup_logging()`，在 lifespan 启动时调用 |
| `utils/` | 通用工具函数，按功能拆文件 |
| `app/weights/` | `.pt` / `.onnx` / `.engine` 权重文件 |

---

## 2.3 多模型串联 + 多引擎目录结构

适用于：多模型流水线、混合 PyTorch / ONNX / TensorRT。

```text
cv-service/                       # 仓库根目录
├── app/
│   ├── __init__.py
│   ├── main.py                   # 入口 + 路由 + lifespan
│   ├── config.py                 # 读取 .env（端口、设备等）
│   ├── config.yaml               # 模型路径、引擎类型、串联顺序
│   ├── schemas.py
│   ├── inference.py              # 统一推理（pytorch / onnx / tensorrt）
│   ├── pipeline.py               # 多模型串联编排
│   ├── models/                   # 各模型业务代码（不是权重）
│   │   ├── __init__.py
│   │   ├── face_detect.py
│   │   ├── face_parse.py
│   │   └── wrinkle.py
│   ├── logger.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image.py
│   │   ├── timer.py
│   │   └── file.py
│   └── weights/
│       ├── face_detect.onnx
│       ├── face_parse.engine
│       └── wrinkle.pt
├── tests/
│   └── test_pipeline.py
├── scripts/
│   ├── export_onnx.py
│   └── build_tensorrt.py
├── requirements.txt
├── .env.example
├── Dockerfile                    # 可选
├── .gitignore
└── README.md
```

| 文件/目录 | 干什么 |
|-----------|--------|
| `inference.py` | 按引擎类型加载 `.pt` / `.onnx` / `.engine` 并执行推理 |
| `pipeline.py` | 串联多个 model：`A → B → C` |
| `models/` | 每个模型的预处理 + 调 inference + 后处理 |
| `config.yaml` | 模型注册表与 pipeline 步骤，换引擎只改配置 |

---

## 2.4 配置约定

### `.env` / `.env.example`（环境变量）

```env
DEVICE=0
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

### `app/config.py`（单模型或多模型共用）

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 .env 读取服务环境配置。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DEVICE: str = "0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    # 单模型时可加：
    MODEL_PATH: str = "app/weights/yolov8n.pt"
    CONF_THRESHOLD: float = 0.5


settings = Settings()
```

### `app/config.yaml`（仅多模型时）

```yaml
models:
  face_detect:
    path: app/weights/face_detect.onnx
    engine: onnx
  face_parse:
    path: app/weights/face_parse.engine
    engine: tensorrt
  wrinkle:
    path: app/weights/wrinkle.pt
    engine: pytorch

pipeline:
  - face_detect
  - face_parse
  - wrinkle
```

---

## 2.5 关键代码示例（单模型 YOLO）

### `app/logger.py`

```python
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """初始化全局日志。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
```

### `app/utils/image.py`

```python
from io import BytesIO

from PIL import Image


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """字节流转 PIL Image。"""
    return Image.open(BytesIO(image_bytes))
```

### `app/detector.py`

```python
from ultralytics import YOLO

from app.config import settings
from app.schemas import Detection, DetectResult


class Detector:
    """YOLO 检测器，与 FastAPI 无关，可单独测试。"""

    def __init__(self, model_path: str, device: str):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image_bytes: bytes) -> DetectResult:
        """对图片字节流执行目标检测。"""
        results = self.model.predict(
            source=image_bytes,
            conf=settings.CONF_THRESHOLD,
            device=self.device,
            verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            detections.append(Detection(
                class_id=int(box.cls),
                class_name=results[0].names[int(box.cls)],
                confidence=float(box.conf),
                bbox=box.xyxy[0].tolist(),
            ))
        return DetectResult(detections=detections, count=len(detections))
```

### `app/schemas.py`

```python
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """单个检测框。"""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")


class DetectResult(BaseModel):
    """检测结果。"""

    detections: list[Detection]
    count: int
```

### `app/main.py`（入口 + 路由合一）

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile

from app.config import settings
from app.detector import Detector
from app.logger import setup_logging
from app.schemas import DetectResult
from app.utils.timer import timer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化日志并加载模型。"""
    setup_logging(settings.LOG_LEVEL)
    app.state.detector = Detector(settings.MODEL_PATH, settings.DEVICE)
    yield


app = FastAPI(title="YOLO Detect Service", lifespan=lifespan)


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResult)
async def detect(request: Request, image: UploadFile = File(...)):
    """上传图片，返回检测结果。"""
    image_bytes = await image.read()
    detector = request.app.state.detector
    with timer("detect"):
        return detector.detect(image_bytes)
```

### 启动

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2.6 多模型关键代码示意

### `app/inference.py`

```python
import numpy as np


def load_model(path: str, engine: str):
    """按引擎类型加载模型。"""
    if engine == "pytorch":
        import torch
        return torch.load(path, map_location="cpu"), engine
    if engine == "onnx":
        import onnxruntime as ort
        return ort.InferenceSession(path), engine
    if engine == "tensorrt":
        # TensorRT 加载逻辑
        ...
    raise ValueError(f"不支持的引擎: {engine}")


def predict(model, engine: str, input_data: np.ndarray) -> np.ndarray:
    """统一推理接口。"""
    if engine == "onnx":
        input_name = model.get_inputs()[0].name
        return model.run(None, {input_name: input_data})[0]
    ...
```

### `app/pipeline.py`

```python
class WrinklePipeline:
    """多模型串联：人脸检测 → 解析 → 皱纹分割。"""

    def __init__(self, models: dict):
        self.models = models

    def run(self, image_bytes: bytes) -> dict:
        """按 config.yaml 中 pipeline 顺序执行。"""
        faces = self.models["face_detect"].run(image_bytes)
        if not faces:
            return {"message": "未检测到人脸"}
        region = self.models["face_parse"].run(image_bytes, faces[0])
        mask = self.models["wrinkle"].run(region)
        return {"wrinkle_mask": mask}
```

---

## 2.7 API 约定

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/detect` 或 `/predict` | 上传图片推理 |

- 图片上传用 `multipart/form-data`（`UploadFile`）
- 小型服务不需要 `/api/v1` 前缀
- 返回结构用 Pydantic 模型，不返回裸 dict

---

## 2.8 必须遵守的底线

| 规则 | 说明 |
|------|------|
| 模型启动时加载 | `lifespan` 中加载，挂 `app.state` |
| 路由不写推理 | 路由只调 detector / pipeline |
| 权重不入库 | `app/weights/` 加入 `.gitignore` |
| 配置不硬编码 | `.env` + `config.py`；多模型加 `config.yaml` |
| 训练独立 | `train.py` 或 `scripts/` 放根目录，不放 `app/` |

---

## 2.9 何时扩展目录

| 触发条件 | 怎么拆 |
|----------|--------|
| 接口超过 3 个 | `main.py` 中的路由 → `routers/` |
| schemas 类型变多 | `schemas.py` → `schemas/` |
| 想对齐 YOLO 教程命名 | `detector.py` → `detectors/yolo.py` |
| 配置/日志文件变多 | `config.py` + `logger.py` → `core/` |
| 有 2 条以上流水线 | `pipeline.py` → `pipelines/` |
| `inference.py` 引擎逻辑膨胀 | 再拆 `engines/` |

在此之前保持当前结构，不要提前过度设计。

---

## 2.10 场景选型

| 场景 | 用哪套 |
|------|--------|
| 1 个模型、1～3 个接口 | **2.2 单模型结构** |
| 多模型串联、混合 .pt/.onnx/.engine | **2.3 多模型结构** |
| 一次性实验、纯训练 | 不建 FastAPI，保持脚本 |

---

## 2.11 调用关系

**单模型：**

```text
main.py（路由） → detector.py → utils/
```

**多模型：**

```text
main.py（路由） → pipeline.py → models/*.py → inference.py
                                      ↓
                                    utils/
```

---

## 2.12 其他 CV 任务适配

| 任务 | 单模型文件 | 多模型目录 |
|------|-----------|-----------|
| 目标检测 | `detector.py` | `models/face_detect.py` |
| 图像分割 | `detector.py`（Segmentor） | `models/segmentor.py` |
| 图像分类 | `detector.py`（Classifier） | `models/classifier.py` |
| OCR | `detector.py`（OcrEngine） | `models/ocr.py` |
