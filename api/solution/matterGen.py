from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import requests
from langchain.tools import StructuredTool


class MatterGenInput(BaseModel):
    batch_size: int = Field(16, description="Number of samples generated per batch, must be a positive integer")
    num_batches: int = Field(1, description="Number of batches to generate, must be a positive integer")
    properties_to_condition_on: Optional[Dict[str, Any]] = Field(
        None, description="Dictionary of conditional properties, e.g., {'dft_mag_density': 0.15}"
    )


def matter_gen_function(
        batch_size: int = 16,
        num_batches: int = 1,
        properties_to_condition_on: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """MatterGen Inorganic Material Structure Generation"""
    invoke_url = "http://mattergen:8000/generate"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model_name": "mattergen_base",
        "batch_size": batch_size,
        "num_batches": num_batches
    }

    if properties_to_condition_on:
        payload["properties_to_condition_on"] = properties_to_condition_on

    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()  # Check HTTP status code
        response_data = response.json()

        # Clean up response data
        if 'data' in response_data and 'generation_time' in response_data['data']:
            del response_data['data']['generation_time']

        return response_data

    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP request error: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request exception: {str(e)}"}
    except ValueError as e:
        return {"error": f"JSON parsing error: {str(e)}"}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}


matter_gen_tool = StructuredTool.from_function(
    func=matter_gen_function,
    name='matterGen_tool',
    description="""
    A tool specialized in crystal material generation and design. 
    Generates material structures based on batch size (batch_size) and number of batches (num_batches).
    Optional parameter properties_to_condition_on can be used to specify conditional properties for generation.
    """,
    args_schema=MatterGenInput
)
