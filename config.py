from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[0]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_CARD_DIR = RAW_DATA_DIR / "archive"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
GENERATED_DATA_DIR = DATA_DIR / "generated"

MODELS_DIR = PROJ_ROOT / "models"