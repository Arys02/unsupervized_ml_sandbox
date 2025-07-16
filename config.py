from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[0]

path_str = str(PROJ_ROOT)
if path_str.startswith("/mnt/x"):
    path_str = path_str[6:]  # 6 est la longueur de "/mnt/x"

PROJ_ROOT = Path(path_str)
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GENERATED_DATA_DIR = DATA_DIR / "generated"

MODELS_DIR = PROJ_ROOT / "models"
KM_DIR = PROJ_ROOT / "models" / "kmeans"
