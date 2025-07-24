from config import ExpertModel
from .alphaFold2 import alphafold2_tool
from .esm3 import esm3_tool
from .evo2 import dna_generate_tool
from .field import field_predict_tool
from .matterGen import matter_gen_tool
from .spectrum import spectrum_predict_tool

tool_map = {

    ExpertModel.DEFAULT: None,
    ExpertModel.ESM3: esm3_tool,
    ExpertModel.EVO2: dna_generate_tool,
    ExpertModel.AlphaFold2: alphafold2_tool,
    ExpertModel.MatterGen: matter_gen_tool,
    ExpertModel.SPECTRUM: spectrum_predict_tool,
    ExpertModel.FIELD: field_predict_tool,

}
