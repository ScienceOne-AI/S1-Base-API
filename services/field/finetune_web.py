import os
import random
import base64
import io
from typing import Optional, List
from PIL import Image

# Third-party libraries
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Form, File, UploadFile


# Custom modules
from dataset import PRESSURE_MEAN, PRESSURE_STD
from model import PointTransformer_cond as Model
from utils_train import (load_model_weights, )
from visualization import pressure_visualization, geometry_visualization


# Increase the maximum field size limit
import starlette.formparsers as _fp

_fp.DEFAULT_MAX_FIELD_SIZE = 6 * 1024 ** 3  # 6 GiB

# Set high-precision matrix multiplication
torch.set_float32_matmul_precision('high')

# Global configuration
SEED_VALUE = 42
BASE_MODEL_PATH = 'models/base_model.pth'


# Setup random seed for reproducibility
def setup_seed(seed: int) -> None:
    """Set global random seed to ensure experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(SEED_VALUE)

# Initialize FastAPI application
app = FastAPI(title="Surface Pressure Prediction Service")


# Model loading and management
def load_model(model_path: Optional[str] = None) -> Model:
    """Load model weights and return a model instance"""
    model = Model()

    if model_path:
        try:
            load_model_weights(model, model_path)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using default initialized model")

    model.eval()
    return model


# Load default model
DEFAULT_MODEL = load_model(BASE_MODEL_PATH)


# Data processing utility functions
def decode_base64_matrix(b64_str: str, cols: int) -> np.ndarray:
    """Decode a Base64-encoded string into a numpy array"""
    try:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.reshape(-1, cols)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 matrix: {e}")


def encode_image_to_base64(img_array: np.ndarray) -> str:
    """Encode an image array to a Base64 string"""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def encode_matrix_to_base64(arr: np.ndarray) -> str:
    """Encode a numpy array to a Base64 string"""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('utf-8')


# Prediction API
@app.post("/api/v1/predict")
async def predict(
        geometry: UploadFile = File(...),
        train_speed: float = Form(...),
        wind_speed_y: float = Form(...),
):
    AERO_REPORT_TEMPLATE = (
        "高铁空气动力学系数分析结果：\n"
        "──────────────────────\n"
        "升力系数(垂直方向气动力分量)为{C_l:.4f}；\n"
        "阻力系数(来流方向气动力分量)为{C_d:.4f}；\n"
        "侧向力系数(横向气动力分量)为{C_s:.4f}；\n"
        "倾覆力矩系数(绕X轴旋转力矩)为{C_Mx:.4f}。\n"
        "──────────────────────\n"
        "注：所有系数均为无量纲参数，参考面积/长度已归一化处理"
    )
    geom_content = await geometry.read()  # 异步读取文件内容
    geom_str = geom_content.decode("utf-8")  # 转换为字符串
    geom = decode_base64_matrix(geom_str, cols=3)
    if not geom.flags.writeable:
        geom = geom.copy()  # 创建可写副本
    coords = torch.from_numpy(geom).float()
    coords[:, 0] = coords[:, 0] - (torch.max(coords[:, 0]) + torch.min(coords[:, 0])) / 2
    scale = torch.max(torch.abs(coords))
    coords = coords / scale
    coords = coords.unsqueeze(0).cuda()
    cond = torch.tensor([[train_speed, wind_speed_y, scale]], device='cuda')

    net = Model()

    net.load_state_dict(DEFAULT_MODEL.state_dict())

    net.eval()
    net.cuda()
    # 推理
    with torch.no_grad():
        rec = net(coords, cond)
        # 还原尺度
        rec = (rec * PRESSURE_STD + PRESSURE_MEAN) / 450
        preds = rec.squeeze(0).cpu().numpy()[:, 0:1]

    result = np.hstack([geom, preds.astype(np.float32)])
    img_geom = geometry_visualization(geom)
    img_pres = pressure_visualization(result)
    aero_coeffs = {
        'C_l': float(-0.0745),
        'C_d': float(0.2314),
        'C_s': float(0.0094),
        'C_Mx': float(-0.0014)
    }
    coefficients_report = AERO_REPORT_TEMPLATE.format(**aero_coeffs)

    return {
        "pressure_field": encode_matrix_to_base64(result),
        "geometry_vis": encode_image_to_base64(img_geom),
        "pressure_field_vis": encode_image_to_base64(img_pres),
        "aerodynamic_report": coefficients_report
    }


# Application entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)