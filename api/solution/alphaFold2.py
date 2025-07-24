from typing import List
import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from requests import RequestException


def predict_protein_structure(
        sequences: List[str],
        algorithm: str,
        skip_template_search: bool,
        bit_score: float,
        databases: List[str],
        e_value: float,
        iterations: int,
        num_predictions_per_model: int,
        relax_prediction: bool
) -> dict | None:
    """
    Sends a request to the AlphaFold2 multimer service to predict protein structure.

    Args:
        sequences: List of protein sequences to predict structure for
        algorithm: Algorithm to use for prediction
        skip_template_search: Whether to skip template search
        bit_score: Bit score threshold
        databases: List of databases to use
        e_value: E-value threshold
        iterations: Number of prediction iterations
        num_predictions_per_model: Number of predictions per model
        relax_prediction: Whether to relax the prediction structure

    Returns:
        JSON response from the API containing structure predictions,
        or None if the request fails
    """
    # Construct request payload
    request_data = {
        'sequences': sequences,
        'algorithm': algorithm,
        'skip_template_search': skip_template_search,
        'bit_score': bit_score,
        'databases': databases,
        'e_value': e_value,
        'iterations': iterations,
        'num_predictions_per_model': num_predictions_per_model,
        'relax_prediction': relax_prediction
    }

    try:
        # Send request to AlphaFold2 API
        response = requests.post(
            url='http://alphafold2-multimer:8000/alphaFold2/multimer/predict-structure-from-sequences',
            json=request_data,
            timeout=300  # 5-minute timeout
        )
        # Raise exception for 4xx/5xx status codes
        response.raise_for_status()
        return response.json()
    except RequestException as error:
        # Return None on any request failure
        return None


class ProteinStructurePredictionInput(BaseModel):
    """Input schema for AlphaFold2 protein structure prediction tool"""
    sequences: List[str] = Field(
        ...,
        description="""
        Sequences of proteins to predict the 3D structure of their complex. Example:
        MPTTIEREFEELDTQRRWQPLYLEIRNESHDNTCCHFWLMVWQQKTKAVVMLNRIVGWTLFFQQNAL,
        MAKVEQVLSLEPQHELKFRGPFTDVVTTNLKLGNPTDRNVCFKVKTTAPRRYCVRPNSGIIDAGASINVSVMKSLSSSLDDTMFFIVGVIIGKIAL
        """
    )
    algorithm: str = Field(
        ...,
        description="Algorithm to use for structure prediction"
    )
    skip_template_search: bool = Field(
        ...,
        description="Whether to skip template search during prediction"
    )
    bit_score: float = Field(
        ...,
        description="Bit score threshold for alignment filtering"
    )
    databases: List[str] = Field(
        ...,
        description="List of databases to search against during prediction"
    )
    e_value: float = Field(
        ...,
        description="E-value threshold for alignment significance"
    )
    iterations: int = Field(
        ...,
        description="Number of prediction iterations to perform"
    )
    num_predictions_per_model: int = Field(
        ...,
        description="Number of predictions to generate per model"
    )
    relax_prediction: bool = Field(
        ...,
        description="Whether to apply structure relaxation to predictions"
    )


# Initialize AlphaFold2 prediction tool
alphafold2_tool = StructuredTool.from_function(
    func=predict_protein_structure,
    name="alphafold2_structure_prediction",
    description="Predicts 3D structure of protein complexes using AlphaFold2",
    args_schema=ProteinStructurePredictionInput
)