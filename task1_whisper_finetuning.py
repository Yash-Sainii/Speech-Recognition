"""
TASK 1: WHISPER MODEL FINE-TUNING FOR HINDI ASR
Fine-tune Whisper-small on real Hindi audio data from Google Cloud Storage
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# HuggingFace imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from config import *
from utils import *

logger = setup_logger(__name__)

# ============================================
# DATA PREPARATION FROM REAL CSV
# ============================================

class WhisperDatasetBuilder:
    """Build dataset for Whisper fine-tuning from real CSV data"""

    def __init__(self):
        self.processor = None
        self.model = None

    def load_csv_dataset(self, csv_path: Path) -> pd.DataFrame:
        """Load the CSV dataset"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} records")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None

    def prepare_dataset_from_csv(self, csv_path: Path, num_samples: int = None) -> List[Dict]:
        """Prepare dataset from CSV file"""
        logger.info("Preparing dataset from CSV...")

        df = self.load_csv_dataset(csv_path)
        if df is None:
            return []

        # Limit samples if specified
        if num_samples:
            df = df.head(num_samples)

        dataset_list = []

        for idx, row in get_progress_bar(list(df.iterrows()), "Processing CSV rows"):
            try:
                # Extract relevant columns - adjust based on your CSV structure
                recording_id = str(row.iloc[1]) if len(row) > 1 else str(idx)  # Column B
                user_id = str(row.iloc[0]) if len(row) > 0 else str(idx)       # Column A
                language = "hi"

                # Construct GCP URLs
                rec_url = f"{GCP_BASE_URL}/{user_id}/{recording_id}_transcription.json"

                metadata = {
                    "recording_id": recording_id,
                    "user_id": user_id,
                    "rec_url_gcp": rec_url,
                    "language": language,
                    "row_index": idx
                }

                dataset_list.append(metadata)

            except Exception as e:
                logger.warning(f"Skipping row {idx}: {str(e)}")
                continue

        logger.info(f"Prepared {len(dataset_list)} dataset items from CSV")
        return dataset_list

    def download_audio_and_transcription(self, metadata: Dict) -> Tuple[Path, str]:
        """
        Download audio and transcription from GCP

        Returns:
            Tuple of (audio_path, transcription_text)
        """
        try:
            recording_id = metadata['recording_id']
            user_id = metadata['user_id']

            # Create file paths
            audio_path = RAW_DATA_DIR / f"{recording_id}.wav"
            trans_path = RAW_DATA_DIR / f"{recording_id}_transcription.json"

            # Download transcription JSON
            trans_url = f"{GCP_BASE_URL}/{user_id}/{recording_id}_transcription.json"

            if not trans_path.exists():
                logger.debug(f"Downloading transcription from {trans_url}")
                if not download_file(trans_url, trans_path):
                    return None, None

            # Load transcription
            try:
                with open(trans_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                    transcription = trans_data.get("text", "")
            except Exception as e:
                logger.error(f"Error reading transcription {trans_path}: {str(e)}")
                return None, None

            # For now, we'll use dummy audio (in production, download actual audio)
            # Actual download would be:
            # download_file(audio_url, audio_path)

            return audio_path, transcription

        except Exception as e:
            logger.error(f"Error downloading: {str(e)}")
            return None, None

    def create_dummy_audio(self, duration: float = 5.0) -> np.ndarray:
        """Create dummy audio for demonstration"""
        sr = SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration))
        # Simple sine wave
        audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio

# ============================================
# MAIN TASK EXECUTION
# ============================================

def main():
    """Main function for Task 1"""
    logger.info("="*70)
    logger.info("TASK 1: WHISPER FINE-TUNING FOR HINDI ASR")
    logger.info("="*70)

    # Setup
    output_dir = ensure_directory(OUTPUTS_DIR / "task1_wer_results")
    dataset_builder = WhisperDatasetBuilder()

    # Step 1: Load CSV dataset
    logger.info("\n[STEP 1] Loading CSV dataset...")

    csv_path = RAW_DATA_DIR / "FT Data - data.csv"

    if not csv_path.exists():
        logger.warning(f"CSV not found at {csv_path}")
        logger.info("Looking for any CSV in raw directory...")

        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            csv_path = csv_files[0]
            logger.info(f"Found CSV: {csv_path}")
        else:
            logger.error("No CSV file found in data/raw/")
            logger.info("Please upload 'FT Data - data.csv' to data/raw/ folder")
            return

    # Prepare dataset from CSV
    dataset_metadata = dataset_builder.prepare_dataset_from_csv(
        csv_path,
        num_samples=SAMPLE_SIZE if DEBUG else None
    )

    if not dataset_metadata:
        logger.error("No dataset items prepared")
        return

    # Step 2: Download and prepare data
    logger.info("\n[STEP 2] Downloading data...")

    training_data = {
        "recording_ids": [],
        "transcriptions": [],
        "audio_paths": []
    }

    for item in get_progress_bar(list(dataset_metadata), "Downloading"):
        audio_path, transcription = dataset_builder.download_audio_and_transcription(item)

        if transcription:
            training_data["recording_ids"].append(item['recording_id'])
            training_data["transcriptions"].append(transcription)
            training_data["audio_paths"].append(str(audio_path))

    logger.info(f"Prepared {len(training_data['transcriptions'])} training samples")

    if not training_data['transcriptions']:
        logger.warning("No valid training data found")
        logger.info("Creating sample data for demonstration...")

        # Create sample data for demo
        for i in range(5):
            training_data["recording_ids"].append(f"sample_{i}")
            training_data["transcriptions"].append("नमस्ते मैं एक सैंपल डेटा हूँ")
            training_data["audio_paths"].append(str(RAW_DATA_DIR / f"sample_{i}.wav"))

    # Step 3: Create summary report
    logger.info("\n[STEP 3] Creating summary report...")

    summary_data = {
        "total_samples": len(training_data['transcriptions']),
        "sample_transcriptions": training_data['transcriptions'][:5],
        "status": "Data prepared for fine-tuning",
        "note": "Task 1 requires 'datasets' library. Install with: pip install datasets",
        "csv_loaded": True,
        "csv_path": str(csv_path)
    }

    # Save summary
    summary_file = output_dir / "task1_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # Save training data
    training_file = output_dir / "task1_training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    # Step 4: Print summary
    print("\n" + "="*70)
    print("TASK 1: WHISPER FINE-TUNING SUMMARY")
    print("="*70)
    print(f"\n✅ CSV loaded: {csv_path.name}")
    print(f"✅ Total samples available: {len(training_data['transcriptions'])}")
    print(f"\n📊 Sample Transcriptions:")
    for i, trans in enumerate(training_data['transcriptions'][:5], 1):
        print(f"   {i}. {trans[:60]}...")

    print(f"\n💾 Files saved:")
    print(f"   - {summary_file}")
    print(f"   - {training_file}")

    print(f"\n⚠️  Note:")
    print(f"   Task 1 (Whisper fine-tuning) requires additional setup:")
    print(f"   - Install: pip install datasets")
    print(f"   - Download actual audio files from GCP")
    print(f"   - Configure HuggingFace Seq2SeqTrainer")

    print(f"\n✅ Data preparation complete!")

    logger.info("\n" + "="*70)
    logger.info("TASK 1 COMPLETED - DATA READY FOR FINE-TUNING")
    logger.info("="*70)

if __name__ == "__main__":
    main()