from langchain.tools import StructuredTool

from typing import Dict, Any
from pydantic import BaseModel, Field
import requests


class SpectrumPredictInput(BaseModel):
    query: str = Field(
        description='User query for spectrum analysis. Example format: "Given multiple spectra, predict which compound the spectra correspond to and give the SMILES of that compound. Please answer strictly in the format ##SMILES:"'
    )


def spectrum_predict(
        query: str
) -> Dict[str, Any]:
    """
    Analyzes spectral data based on the provided query to predict chemical compounds and their SMILES representations.

    Args:
        query (str): User's question or instruction regarding the spectral data analysis.
                     Should follow the format specified in the description.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - content: The raw response content from the AI model.
            - smiles: The extracted SMILES string if present in the response, otherwise an empty string.
            - error: An error message if an exception occurs during processing.
    """
    invoke_url = 'http://spectrum-service:5002/v1/chat/completions'
    system_prompt = "You are a professional chemist specialized in spectral analysis. Given spectral data descriptions, analyze the data to identify functional groups and deduce the corresponding chemical compound. Respond with the SMILES notation strictly formatted as ##SMILES: followed by the SMILES string."

    payload = {
        "model": "test",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
    }

    try:
        response = requests.post(invoke_url, json=payload)
        response.raise_for_status()
        result_json = response.json()
        content = result_json.get('choices', [{}])[0].get('message', {}).get('content', '')

        result = {
            "content": content,
            "smiles": ""
        }

        if "##SMILES:" in content:
            result["smiles"] = content.split("##SMILES:")[1].strip()

        return result

    except requests.exceptions.RequestException as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}


spectrum_predict_tool = StructuredTool.from_function(
    func=spectrum_predict,
    name='spectrum_predict',
    description="""
    Predicts chemical structures based on spectral data provided in the user's prompt. 
    Example usage:
    "Given multiple spectra, predict which compound the spectra correspond to and give the SMILES of that compound. Please answer strictly in the format ##SMILES:"
    "Given the crystal diffraction spectrum, predict which crystal system does this spectrum represent. Please answer strictly in the format ##Crystal System:"
    """,
    args_schema=SpectrumPredictInput,
)