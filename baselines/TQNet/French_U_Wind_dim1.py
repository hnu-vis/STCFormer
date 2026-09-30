from baselines.weather_baseline_config import build_weather_cfg
from .arch import TQNet
from .model_config import model_params

CFG = build_weather_cfg("French_U_Wind_dim1", TQNet, model_params)
