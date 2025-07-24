import base64
import httpx
import io
import numpy as np
from PIL import Image
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


def encode_geometry_from_url(url: str) -> tuple[str, int]:
    """
    Loads point cloud data from a remote URL and encodes it as base64.

    Args:
        url: URL of the geometry file containing space-separated 3D coordinates.

    Returns:
        Tuple containing:
        - Base64-encoded string of the binary float32 matrix
        - Number of rows (points) in the matrix
    """
    try:
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.content

        # Parse text content into Nx3 float32 matrix (discarding any additional columns)
        geom = np.genfromtxt(io.BytesIO(content), delimiter=" ", dtype=np.float32)[:, :3]
        raw_bytes = geom.tobytes()
        return base64.b64encode(raw_bytes).decode("utf-8"), geom.shape[0]
    except Exception as e:
        raise RuntimeError(f"Failed to process geometry from URL: {str(e)}")


def decode_matrix(b64_str: str, rows: int, cols: int) -> np.ndarray:
    """
    Decodes a base64-encoded binary string into a numpy float32 matrix.

    Args:
        b64_str: Base64-encoded string of binary float32 data
        rows: Number of rows in the matrix
        cols: Number of columns in the matrix

    Returns:
        Numpy array with shape (rows, cols)
    """
    try:
        raw_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(raw_bytes, dtype=np.float32)
        return arr.reshape(rows, cols)
    except Exception as e:
        raise ValueError(f"Failed to decode matrix: {str(e)}")


def decode_image(b64_str: str, save_path: str = None) -> Image.Image:
    """
    Decodes a base64-encoded image and optionally saves it to disk.

    Args:
        b64_str: Base64-encoded image data
        save_path: Optional file path to save the image

    Returns:
        PIL Image object
    """
    try:
        img_data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_data))
        if save_path:
            img.save(save_path)
        return img
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


class MechanicsCalculateInput(BaseModel):
    """
    Input schema for aerodynamics calculation tool.
    """
    matrix_url: str = Field(
        description="URL of the 3D geometry file describing the spatial distribution of physical points"
    )
    velocity: Dict[str, float] = Field(
        description="Velocity vector (x, y, z components) representing the speed of the object in 3D space (m/s)",
        examples=[{"x": 75, "y": 0, "z": 0}]
    )
    wind: Dict[str, float] = Field(
        description="Wind velocity vector (x, y, z components) representing environmental wind conditions in 3D space (m/s)",
        examples=[{"x": 0, "y": 0, "z": 0}]
    )


def mechanics_calculate(
        matrix_url: str,
        velocity: dict,
        wind: dict
) -> Dict[str, Any]:
    """
    Performs aerodynamics calculations based on provided 3D geometry and flow conditions.

    Args:
        matrix_url: URL of the geometry file
        velocity: Velocity vector dictionary with keys 'x', 'y', 'z'
        wind: Wind vector dictionary with keys 'x', 'y', 'z'

    Returns:
        Dictionary containing aerodynamics coefficients and visualization data
        or error information if the request fails
    """
    invoke_url = "http://field:8000/api/v1/predict"

    try:
        # Encode geometry data from URL
        geometry_b64, N = encode_geometry_from_url(matrix_url)

        # Prepare request payload
        payload = {
            "train_speed": velocity.get("x"),  # Train speed in x-direction
            "wind_speed_y": wind.get("y"),  # Lateral wind speed
            "wind_speed_z": wind.get("z"),  # Vertical wind speed component
        }

        # Send request to aerodynamics API
        with httpx.Client(timeout=6000) as client:
            files = {
                "geometry": ("geometry.txt", geometry_b64.encode('utf-8'), "text/plain"),
            }
            response = client.post(invoke_url, data=payload, files=files)

        # Validate and return response
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


# Initialize LangChain tool
field_predict_tool = StructuredTool.from_function(
    func=mechanics_calculate,
    name='field_predict',
    description="""
    Performs aerodynamics calculations based on provided 3D geometry, velocity vector, and wind vector.
    Returns aerodynamic coefficients and pressure field visualization for high-speed trains.
    """,
    args_schema=MechanicsCalculateInput
)