# Standard library imports
import os
from contextlib import asynccontextmanager

# Third-party imports
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Literal
import uvicorn

# Local application imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

# Constants
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY_")
MAX_STEPS = 50


def validate_sequence(sequence: str) -> None:
    """
    Validate the input protein sequence.

    Args:
        sequence (str): Protein sequence to validate.

    Raises:
        ValueError: If the sequence is empty or contains invalid characters.
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    invalid_chars = [c for c in sequence if c.upper() not in VALID_AMINO_ACIDS]
    if invalid_chars:
        raise ValueError(f"Invalid characters in sequence: {invalid_chars}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for FastAPI application to handle model loading and cleanup.
    """
    # Set infrastructure provider
    os.environ['INFRA_PROVIDER'] = 'local'

    try:
        # Load ESM3 model to GPU if available
        global model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ESM3.from_pretrained("esm3-sm-open-v1").to(device)
        yield
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    finally:
        # Clean up model resources
        if 'model' in globals():
            del model
            torch.cuda.empty_cache()


# Initialize FastAPI application
app = FastAPI(
    title="ESM3 Protein Generation API",
    description="Protein sequence generation, structure prediction, and loop design",
    lifespan=lifespan
)


# Request Models
class BaseGenerationRequest(BaseModel):
    sequence: str
    num_steps: Optional[int] = 20
    temperature: Optional[float] = 0.7


class RoundTripRequest(BaseGenerationRequest):
    sequence_steps: Optional[int] = 8
    structure_steps: Optional[int] = 8


class StructureToSequenceResponse(BaseModel):
    sequence: str


@app.post("/generate-sequence",
          summary="Complete protein sequence",
          description="Generate a full amino acid sequence from a partial input sequence")
async def generate_sequence(request: BaseGenerationRequest):
    protein = ESMProtein(sequence=request.sequence)
    protein = model.generate(
        protein,
        GenerationConfig(
            track="sequence",
            num_steps=request.num_steps,
            temperature=request.temperature
        )
    )
    return JSONResponse(
        content={"sequence": protein.sequence, 'status': 1},
        status_code=200
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
