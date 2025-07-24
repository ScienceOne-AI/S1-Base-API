import requests
from requests import RequestException

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Valid amino acid characters (standard 20 amino acids + placeholder '_')
VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY_"
# Maximum allowed generation steps (as configuration reference)
MAX_STEPS = 50


def execute(sequence: str, left_length: int, right_length: int, num_steps: int = 15, temperature: float = 0.6) -> dict:
    """
    Generates or completes protein sequences using the ESM3 model via API call.

    Args:
        sequence: Core protein sequence to be extended (1-300 characters)
        left_length: Number of residues to add to the left (N-terminus), ≥0
        right_length: Number of residues to add to the right (C-terminus), ≥0
        num_steps: Number of generation iterations (affects sequence completeness)
        temperature: Sampling temperature (controls randomness, 0.1-2.0)

    Returns:
        Dictionary containing generated sequence data from ESM3 API,
        or None if the request fails.
    """
    # Construct padded sequence with placeholders for left/right extension
    padded_sequence = f'{"_" * left_length}{sequence}{"_" * right_length}'

    request_payload = {
        'sequence': padded_sequence,
        'num_steps': num_steps,
        'temperature': temperature
    }

    url = 'http://esm3:8000/generate-sequence'

    try:
        # Send POST request to ESM3 service with timeout protection
        response = requests.post(
            url,
            json=request_payload,
            timeout=300  # 5-minute timeout for long-running generation tasks
        )
        response.raise_for_status()  # Raise exception for 4xx/5xx status codes
        return response.json()
    except RequestException as e:
        print(f"ESM3 API request failed: {str(e)}")  # In production, use proper logging
        return None


class ProteinSequenceInput(BaseModel):
    """Input schema for ESM3 protein sequence generation/extension"""

    sequence: str = Field(
        min_length=1,
        max_length=300,
        description="Protein sequence to predict or complete (e.g., QATSLRILNNGHAFNVEFDDSQDKAVL)"
    )

    left_length: int = Field(
        default=50,
        ge=0,
        description="Number of residues to add to the left (N-terminus), default 50"
    )

    right_length: int = Field(
        default=50,
        ge=0,
        description="Number of residues to add to the right (C-terminus), default 50"
    )

    num_steps: int = Field(
        default=15,
        ge=1,
        le=MAX_STEPS,
        description=f"Number of generation steps (1 to {MAX_STEPS})"
    )

    temperature: float = Field(
        default=0.6,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.1 = deterministic, 2.0 = highly random)"
    )


esm3_tool: StructuredTool = StructuredTool.from_function(
    func=execute,
    name='esm3_protein_generator',
    description="Generates or completes protein sequences using ESM3 language model (use for protein sequence prediction/extension)",
    args_schema=ProteinSequenceInput
)
