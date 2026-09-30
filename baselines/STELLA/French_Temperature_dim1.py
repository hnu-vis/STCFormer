from baselines.weather_baseline_config import build_weather_cfg
from .arch import STELLA
from .model_config import model_params

CFG = build_weather_cfg("French_Temperature_dim1", STELLA, model_params)
