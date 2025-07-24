import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, Field
import shutil
import tempfile
import zipfile
import base64
import io

# Import MatterGen core components
from repo.common.data.types import TargetProperty
from repo.common.utils.data_classes import MatterGenCheckpointInfo
from repo.generator import CrystalGenerator
from repo.common.utils.eval_utils import load_structures
from repo.evaluation.evaluate import evaluate
from repo.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)
from repo.common.utils.globals import get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global request status tracking
# Structure: {request_id: {"status": "running|completed|failed", "start_time": timestamp, "end_time": timestamp, "error": error_message}}
request_status = {}
# Regularly clean up old completed requests (older than 24 hours)
MAX_STATUS_AGE = 24 * 60 * 60  # 24 hours (seconds)

# Create FastAPI application
app = FastAPI(
    title="MatterGen API",
    description="API for generating and evaluating inorganic materials using MatterGen.",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In a production environment, this should be limited to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class GenerationRequest(BaseModel):
    model_name: str = Field("mattergen_base", description="Name of the pre-trained model or path to a custom model")
    batch_size: int = Field(16, description="Number of samples to generate per batch")
    num_batches: int = Field(1, description="Number of batches to generate")
    properties_to_condition_on: Optional[Dict[str, Any]] = Field(None,
                                                                 description="Dictionary of properties to condition on, e.g., {'dft_mag_density': 0.15}")
    diffusion_guidance_factor: Optional[float] = Field(None,
                                                       description="Diffusion guidance factor (gamma). Set to 0 for unconditional generation if supported by the model")
    record_trajectories: bool = Field(False, description="Whether to record and output generation trajectories")
    use_cache: bool = Field(True, description="Whether to use caching, defaults to True")


# Standard API response model
class APIResponse(BaseModel):
    code: int = Field(200, description="Status code, 200 for success, others for errors")
    message: str = Field("Success", description="Response message, success or error description")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


# Standard error response model
class ErrorResponse(BaseModel):
    code: int = Field(..., description="Error status code")
    message: str = Field(..., description="Error description")
    details: Optional[str] = Field(None, description="Detailed error information")


# Generation response model
class GenerationResponse(BaseModel):
    # File URLs
    crystals_cif_url: Optional[str] = Field(None, description="MinIO URL for CIF structure files")
    crystals_extxyz_url: Optional[str] = Field(None, description="MinIO URL for EXTXYZ structure files")
    trajectories_url: Optional[str] = Field(None, description="MinIO URL for trajectory files (if requested)")
    # Available file formats
    file_formats: List[str] = Field([], description="List of available file formats")
    total_structures: int = Field(0, description="Total number of generated structures")


def upload_file(filepath, bucket_name=None, folder_path=None, filename=None):
    """
    Upload a file directly to MinIO storage.

    Args:
        filepath: Path to the file to upload
        bucket_name: Name of the bucket to upload to
        folder_path: Path within the bucket to store the file
        filename: Name to give the file in storage

    Returns:
        Full URL of the uploaded file
    """
    # Get MinIO configuration
    minio_url = os.getenv("MINIO_URL", )
    minio_access_key = os.getenv("MINIO_ACCESS_KEY",)
    minio_secret_key = os.getenv("MINIO_SECRET_KEY",)
    minio_bucket = os.getenv("MINIO_BUCKET_NAME",)
    minio_secure = os.getenv("MINIO_SECURE", "true").lower() == "true"

    # Use default bucket if not provided
    if bucket_name is None:
        bucket_name = minio_bucket

    # Prepare file name and object name
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d/%H%M%S_")

    if filename is None:
        filename = os.path.basename(filepath)

    object_name = f"{folder_path}/{formatted_time}{filename}" if folder_path else f"{formatted_time}{filename}"

    # Determine content type based on file extension
    content_type = "application/octet-stream"  # Default type
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext == '.zip':
        content_type = 'application/zip'
    elif file_ext == '.cif':
        content_type = 'chemical/x-cif'
    elif file_ext == '.extxyz' or file_ext == '.xyz':
        content_type = 'text/plain'
    elif file_ext == '.json':
        content_type = 'application/json'
    elif file_ext == '.txt':
        content_type = 'text/plain'

    # Create MinIO client
    try:
        client = Minio(
            minio_url,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure
        )

        # Check if bucket exists, create if not
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Created bucket: {bucket_name}")

        # Upload file
        logger.info(f"Uploading file {filepath} to {bucket_name}/{object_name}")
        client.fput_object(
            bucket_name,
            object_name,
            filepath,
            content_type=content_type
        )

        # Construct file URL
        protocol = "https" if minio_secure else "http"
        file_url = f"{protocol}://{minio_url}/{bucket_name}/{object_name}"

        return file_url

    except S3Error as e:
        logger.error(f"MinIO error: {e}")
        raise Exception(f"Error uploading file to MinIO: {e}")


def file_to_base64(file_path: Path) -> str:
    """Convert file content to a base64 encoded string (retained for backward compatibility)"""
    if not file_path.exists():
        return None
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_cif_files_from_zip(zip_path: Path) -> List[Dict[str, str]]:
    """Extract CIF files from a ZIP file and return a list of their contents"""
    if not zip_path.exists():
        return None

    cif_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.endswith('.cif'):
                with zip_ref.open(filename) as f:
                    content = f.read().decode('utf-8')
                    cif_files.append({
                        "name": filename,
                        "content": content
                    })
    return cif_files


def read_text_file(file_path: Path) -> str:
    """Read the content of a text file"""
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        return f.read()


# --- API Endpoints ---

@app.post("/generate", response_model=APIResponse)
async def generate_materials(
        request: GenerationRequest,
        background_tasks: BackgroundTasks
):
    """
    Generate materials using the MatterGen model based on the provided parameters.

    Args:
        request: Generation request parameters
        background_tasks: Background task manager for cleanup

    Returns:
        APIResponse containing URLs to generated files and metadata
    """
    # Create a unique request ID
    request_id = str(uuid.uuid4())

    try:
        # Prepare request data for caching (excluding use_cache field)
        cache_request_data = request.dict(exclude={'use_cache'})
        # Create temporary directory
        results_dir = tempfile.mkdtemp(prefix=f"mattergen_api_{request_id}_")

        # Update request status
        if request_id not in request_status:
            request_status[request_id] = {
                "status": "running",
                "start_time": time.time(),
                "command": f"Calling generate_materials_directly to generate materials",
                "session_id": str(uuid.uuid4())[:8]
            }

        # Directly call the generation function
        await generate_materials_directly(
            output_path=results_dir,
            model_name=request.model_name,
            batch_size=request.batch_size,
            num_batches=request.num_batches,
            properties_to_condition_on=request.properties_to_condition_on,
            diffusion_guidance_factor=request.diffusion_guidance_factor,
            record_trajectories=request.record_trajectories
        )

        # Prepare response with generated files
        generation_response = GenerationResponse(file_formats=[], total_structures=0)

        # Process CIF format crystal structures
        cif_zip = Path(results_dir) / "generated_crystals_cif.zip"
        if cif_zip.exists():
            # Upload CIF file
            generation_response.crystals_cif_url = upload_file(
                str(cif_zip),
                folder_path="mattergen",
                filename=f"{request_id}_generated_crystals_cif.zip"
            )
            generation_response.file_formats.append("cif")

            # Unzip and count the number of structures
            with zipfile.ZipFile(cif_zip, 'r') as zip_ref:
                cif_files = [f for f in zip_ref.namelist() if f.endswith('.cif')]
                generation_response.total_structures = len(cif_files)

        # Process extxyz format crystal structures
        extxyz_file = Path(results_dir) / "generated_crystals.extxyz"
        if extxyz_file.exists():
            # Upload EXTXYZ file
            generation_response.crystals_extxyz_url = upload_file(
                str(extxyz_file),
                folder_path="mattergen",
                filename=f"{request_id}_generated_crystals.extxyz"
            )
            generation_response.file_formats.append("extxyz")

            # If structure count not determined from CIF files, estimate from EXTXYZ file
            if not generation_response.total_structures:
                with open(extxyz_file, 'r') as f:
                    content = f.read()
                    generation_response.total_structures = content.count("ITEM: TIMESTEP")

        # Process trajectory data (if requested)
        if request.record_trajectories:
            trajectories_zip = Path(results_dir) / "generated_trajectories.zip"
            if trajectories_zip.exists():
                generation_response.trajectories_url = upload_file(
                    str(trajectories_zip),
                    folder_path="mattergen",
                    filename=f"{request_id}_generated_trajectories.zip"
                )
                generation_response.file_formats.append("trajectories")

        # Check if any output files were generated
        if not generation_response.file_formats:
            raise HTTPException(status_code=500, detail="Generation completed but no output files were found")

        # Prepare response data
        response_data = {
            **generation_response.dict(exclude_none=True),
            "request_id": request_id,
            "from_cache": False,
            "generation_time": datetime.datetime.now().isoformat()
        }

        # Update request status to completed
        if request_id in request_status:
            request_status[request_id].update({
                "status": "completed",
                "end_time": time.time()
            })

        # Safely delete the temporary directory
        def safe_cleanup(temp_dir_path):
            try:
                if os.path.exists(temp_dir_path):
                    shutil.rmtree(temp_dir_path)
                    logger.info(f"Cleaned up temporary directory: {temp_dir_path}")
            except Exception as e:
                logger.error(f"Failed to cleanup temporary directory: {temp_dir_path}, Error: {str(e)}")

        # Add background task to cleanup temporary directory
        background_tasks.add_task(safe_cleanup, str(results_dir))

        # Create standard response
        response = APIResponse(
            code=200,
            message="Materials generation completed successfully",
            data=response_data
        )

        return response

    except Exception as e:
        # Update request status to failed
        if request_id in request_status:
            request_status[request_id].update({
                "status": "failed",
                "end_time": time.time(),
                "error": str(e)
            })

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code=500,
                message="An unexpected error occurred during generation",
                details=str(e)
            ).dict()
        )


# --- Core Functionality ---

async def generate_materials_directly(
        output_path: str,
        model_name: str = "mattergen_base",
        batch_size: int = 16,
        num_batches: int = 1,
        properties_to_condition_on: Optional[Dict[str, Any]] = None,
        diffusion_guidance_factor: Optional[float] = None,
        record_trajectories: bool = False,
) -> None:
    """
    Generate materials directly using the MatterGen core library without command-line invocation.

    Args:
        output_path: Path to the output directory
        model_name: Name or path of the model
        batch_size: Batch size for generation
        num_batches: Number of batches to generate
        properties_to_condition_on: Properties to condition the generation on
        diffusion_guidance_factor: Diffusion guidance factor
        record_trajectories: Whether to record trajectories during generation
    """
    logger.info(f"Starting direct material generation, output directory: {output_path}")

    try:
        # Ensure output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Initialize properties
        properties_to_condition_on = properties_to_condition_on or {}
        logger.info(f"Conditioning properties: {properties_to_condition_on}")

        # Check if it's a pre-trained model name or custom model path
        if model_name.startswith("/") or Path(model_name).exists():
            logger.info(f"Using custom model path: {model_name}")
            checkpoint_info = MatterGenCheckpointInfo(
                model_path=Path(model_name).resolve(),
                load_epoch="last",
                config_overrides=[
                    "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
                ],
                strict_checkpoint_loading=True,
            )
        else:
            logger.info(f"Using pre-trained model: {model_name}")
            checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
                model_name,
                config_overrides=[
                    "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
                ]
            )

        # Create generator
        generator = CrystalGenerator(
            checkpoint_info=checkpoint_info,
            properties_to_condition_on=properties_to_condition_on,
            batch_size=batch_size,
            num_batches=num_batches,
            record_trajectories=record_trajectories,
            diffusion_guidance_factor=(
                diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
            ),
        )

        # Execute generation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: generator.generate(output_dir=Path(output_path))
        )

    except Exception as e:
        error_msg = f"Error occurred during material generation: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


if __name__ == "__main__":
    import uvicorn


    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, preparing to shut down...")
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host="0.0.0.0", port=8000)