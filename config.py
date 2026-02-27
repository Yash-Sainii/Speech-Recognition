"""
Configuration file for HindiASR_ResearchProject
All settings and parameters are defined here
"""

import os
import logging
from pathlib import Path



def setup_logger(name: str) -> logging.Logger:
    """Setup logger for the project"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logger(__name__)

# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# GOOGLE CLOUD STORAGE CONFIGURATION
# ============================================

GCP_BASE_URL = "https://storage.googleapis.com/upload_goai"

# ============================================
# WHISPER MODEL CONFIGURATION (TASK 1)
# ============================================

WHISPER_MODEL_NAME = "small"
DEVICE = "cpu"  # Use "cuda" for GPU, "cpu" for CPU
LANGUAGE = "hi"

# Fine-tuning parameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 1
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

# Model saving
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 3

# ============================================
# AUDIO PROCESSING CONFIGURATION
# ============================================

SAMPLE_RATE = 16000
AUDIO_MAX_DURATION = 30
AUDIO_MIN_DURATION = 0.5

# Audio preprocessing
NORMALIZE_AUDIO = True
REMOVE_SILENCE = True
SILENCE_THRESHOLD = 0.02

# ============================================
# TASK 2: DISFLUENCY DETECTION
# ============================================

TARGET_DISFLUENCIES = [
    "filler",
    "repetition",
    "false_start",
    "prolongation",
    "hesitation",
    "restart",
]

HINDI_FILLERS = [
    "अह",
    "उम्म",
    "एर्म",
    "हाँ",
    "नहीं",
    "तो",
]

# ============================================
# TASK 3: SPELLING CORRECTION
# ============================================

VALID_DEVANAGARI_CHARS = [
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
    'क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'ळ',
    'व', 'श', 'ष', 'स', 'ह',
    '\u093E', '\u093F', '\u0940', '\u0941', '\u0942',
    '\u0902', '\u0903',
    '\u094D',
]

MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 50

# ============================================
# TASK 4: LATTICE-BASED WER
# ============================================

ALIGNMENT_UNIT = "word"
USE_MODEL_AGREEMENT = True
MODEL_AGREEMENT_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.5

CASE_SENSITIVE = False
REMOVE_PUNCTUATION = True
REMOVE_MULTIPLE_SPACES = True

# ============================================
# EVALUATION CONFIGURATION
# ============================================

FLEURS_LANGUAGE = "hi"
FLEURS_SPLIT = "test"

WER_METRIC = "word_error_rate"
CER_METRIC = "character_error_rate"

# ============================================
# OUTPUT CONFIGURATION
# ============================================

CSV_ENCODING = "utf-8-sig"
CSV_INDEX = False

AUDIO_FORMAT = "wav"
AUDIO_BITRATE = "192k"

# ============================================
# LOGGING CONFIGURATION
# ============================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "project.log"

# ============================================
# PROCESSING PARAMETERS
# ============================================

NUM_WORKERS = 4
CHUNK_SIZE = 100

MAX_CACHE_SIZE = 5000
ENABLE_MEMORY_OPTIMIZATION = True

# ============================================
# VALIDATION RULES
# ============================================

EXPECTED_FIELDS = ["user_id", "recording_id", "language", "duration",
                   "rec_url_gcp", "transcription_url", "metadata_url"]

REQUIRED_FIELDS = ["recording_id", "rec_url_gcp", "transcription_url"]

# ============================================
# DEBUG MODE
# ============================================

DEBUG = True  # Set to True for testing with small data
VERBOSE = True
SAMPLE_SIZE = 10  # Process only 10 items for testing
