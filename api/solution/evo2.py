import requests
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class DnaGenerateInput(BaseModel):
    """Input schema for DNA sequence generation tool"""
    sequence: str = Field(
        min_length=1, max_length=500,
        description="DNA sequence, e.g., TCCATCTGAGGTACCGGGTTCATCTCACTAGGGAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGGCGAGGCATTGCCTCACCTGGGAAGCGCAAGGGGTCAGGGAGTTCCCTTTCCGA"
    )
    num_tokens: int = Field(default=100, ge=1, le=300, description="Length of the DNA sequence to predict")
    temperature: float = Field(
        default=0.7, gt=0.01, le=1.3,
        description="Randomness control during sampling. Lower values (<1.0) produce less random results, higher values (>1.0) increase randomness"
    )
    top_k: int = Field(
        default=3, gt=0, le=6,
        description="Number of highest probability tokens to consider. Set to 1 for most likely token, higher values increase diversity"
    )
    top_p: float = Field(
        default=1.0, gt=0, le=1.0,
        description="Nucleus sampling threshold. Filters tokens whose cumulative probability exceeds this value"
    )


def dna_generate(
        sequence: str,
        num_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 3,
        top_p: float = 1.0
) -> dict:
    """
    Generate a DNA sequence prediction based on the input sequence using AI model.

    Args:
        sequence: Input DNA sequence (min 1, max 500 characters)
        num_tokens: Number of tokens to generate (1-300)
        temperature: Controls randomness (0.01-1.3)
        top_k: Number of highest probability tokens to consider (1-6)
        top_p: Nucleus sampling threshold (0.0-1.0)

    Returns:
        dict: Generated DNA sequence and metadata

    Raises:
        HTTPException: If the API request fails
    """
    url = 'http://evo2:8000/biology/arc/evo2/generate'

    headers = {
        'Content-Type': 'application/json',
    }

    req_body = {
        "sequence": sequence,
        "num_tokens": num_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "enable_logits": False,
        "enable_sampled_probs": True
    }

    try:
        response = requests.post(url, headers=headers, json=req_body)
        response.raise_for_status()  # Raise exception for 4xx/5xx status codes
        result = response.json()
        result["input_sequence"] = sequence
        return result

    except requests.exceptions.HTTPError as http_err:
        error_msg = f"HTTP error occurred: {http_err}"
        raise ValueError(f"API request failed: {response.status_code} - {error_msg}") from http_err

    except requests.exceptions.RequestException as req_err:
        error_msg = f"Request exception occurred: {req_err}"
        raise ValueError(f"Network error: {error_msg}") from req_err

    except ValueError as json_err:
        error_msg = f"Failed to parse API response: {json_err}"
        raise ValueError(f"Invalid API response format: {error_msg}") from json_err


# Initialize the tool with proper type hints
dna_generate_tool: StructuredTool = StructuredTool.from_function(
    func=dna_generate,
    name='dna_predict',
    description="Predict the next segment of a DNA sequence based on the input sequence",
    args_schema=DnaGenerateInput
)