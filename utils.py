"""
Utility functions for HindiASR_ResearchProject
Common functions used across all tasks
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf
from tqdm import tqdm
import requests
from urllib.parse import urlparse
import hashlib

from config import *


# ============================================
# LOGGING SETUP
# ============================================

def setup_logger(name: str) -> logging.Logger:
    """Setup logger for the project"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Create handlers
    handler = logging.FileHandler(LOG_FILE)
    handler.setLevel(LOG_LEVEL)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logger(__name__)


# ============================================
# FILE OPERATIONS
# ============================================

def ensure_directory(directory_path: Path) -> Path:
    """Ensure directory exists, create if not"""
    directory_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {directory_path}")
    return directory_path


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download file from URL with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded: {url} -> {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False


def load_json(json_path: Path) -> Dict:
    """Load JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {json_path}: {str(e)}")
        return {}


def save_json(data: Dict, output_path: Path) -> bool:
    """Save data to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {output_path}: {str(e)}")
        return False


# ============================================
# AUDIO OPERATIONS
# ============================================

def load_audio(audio_path: Path, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio file"""
    try:
        audio, sr = librosa.load(str(audio_path), sr=sr)
        logger.debug(f"Loaded audio: {audio_path}, shape: {audio.shape}")
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio from {audio_path}: {str(e)}")
        return None, None


def save_audio(audio: np.ndarray, output_path: Path, sr: int = SAMPLE_RATE) -> bool:
    """Save audio file"""
    try:
        sf.write(str(output_path), audio, sr)
        logger.info(f"Saved audio: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save audio to {output_path}: {str(e)}")
        return False


def get_audio_duration(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Get duration of audio in seconds"""
    return len(audio) / sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range"""
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def remove_silence(audio: np.ndarray, sr: int = SAMPLE_RATE,
                   threshold_db: float = SILENCE_THRESHOLD) -> np.ndarray:
    """Remove silence from audio"""
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Get time frames above threshold
    mask = np.mean(S_db, axis=0) > threshold_db

    # Convert frame indices to sample indices
    frames = np.where(mask)[0]
    if len(frames) == 0:
        return audio

    start_frame = frames[0]
    end_frame = frames[-1]

    start_sample = librosa.frames_to_samples(start_frame)
    end_sample = librosa.frames_to_samples(end_frame)

    return audio[start_sample:end_sample]


def segment_audio(audio: np.ndarray, start_time: float, end_time: float,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Segment audio between two timestamps"""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    return audio[start_sample:end_sample]


# ============================================
# TEXT OPERATIONS
# ============================================

def is_valid_devanagari(text: str) -> bool:
    """Check if text contains valid Devanagari characters"""
    devanagari_count = 0
    for char in text:
        if ord(char) >= 0x0900 and ord(char) <= 0x097F:
            devanagari_count += 1

    return devanagari_count > 0


def has_english_word(text: str) -> bool:
    """Check if text contains English words"""
    english_count = sum(1 for char in text if char.isascii() and char.isalpha())
    return english_count > 0


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra spaces
    text = ' '.join(text.split())
    # Remove special characters but keep Devanagari
    text = ''.join(char for char in text if char.isalnum() or
                   (ord(char) >= 0x0900 and ord(char) <= 0x097F) or
                   char in ' \t\n')
    return text


# ============================================
# DATA OPERATIONS
# ============================================

def create_dataframe(data_list: List[Dict]) -> pd.DataFrame:
    """Create pandas DataFrame from list of dicts"""
    return pd.DataFrame(data_list)


def save_csv(df: pd.DataFrame, output_path: Path) -> bool:
    """Save DataFrame to CSV"""
    try:
        df.to_csv(output_path, encoding=CSV_ENCODING, index=CSV_INDEX)
        logger.info(f"Saved CSV: {output_path}, shape: {df.shape}")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV to {output_path}: {str(e)}")
        return False


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV file"""
    try:
        df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
        logger.info(f"Loaded CSV: {csv_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV from {csv_path}: {str(e)}")
        return None


# ============================================
# WER CALCULATION
# ============================================

from jiwer import wer, cer


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    try:
        return wer(reference, hypothesis)
    except Exception as e:
        logger.error(f"WER calculation failed: {str(e)}")
        return 1.0


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate"""
    try:
        return cer(reference, hypothesis)
    except Exception as e:
        logger.error(f"CER calculation failed: {str(e)}")
        return 1.0


def calculate_metrics(references: List[str], hypotheses: List[str]) -> Dict:
    """Calculate WER and CER for multiple pairs"""
    if len(references) != len(hypotheses):
        logger.error("Number of references and hypotheses must match")
        return {}

    wers = [calculate_wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    cers = [calculate_cer(ref, hyp) for ref, hyp in zip(references, hypotheses)]

    return {
        'wer': np.mean(wers),
        'cer': np.mean(cers),
        'wer_std': np.std(wers),
        'cer_std': np.std(cers),
        'wers': wers,
        'cers': cers
    }


# ============================================
# BATCH PROCESSING
# ============================================

def batch_process(items: List, batch_size: int = CHUNK_SIZE) -> List[List]:
    """Split items into batches"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


# ============================================
# VALIDATION
# ============================================

def validate_metadata(metadata: Dict) -> bool:
    """Validate if metadata has required fields"""
    for field in REQUIRED_FIELDS:
        if field not in metadata:
            logger.warning(f"Missing required field: {field}")
            return False
    return True


def validate_audio_duration(duration: float) -> bool:
    """Check if audio duration is within valid range"""
    return AUDIO_MIN_DURATION <= duration <= AUDIO_MAX_DURATION


# ============================================
# PROGRESS TRACKING
# ============================================

def get_progress_bar(items: List, description: str = "Processing"):
    """Return tqdm progress bar"""
    return tqdm(items, desc=description, total=len(items))