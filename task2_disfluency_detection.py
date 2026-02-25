"""
TASK 2: DISFLUENCY DETECTION & SEGMENTATION
Detect fillers, repetitions, false starts, prolongations from audio
Create segmented audio clips and CSV with metadata
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import re
from dataclasses import dataclass, asdict

import librosa
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

from config import *
from utils import *

logger = setup_logger(__name__)


# ============================================
# DISFLUENCY PATTERNS & DETECTION
# ============================================

@dataclass
class Disfluency:
    """Data class for disfluency"""
    recording_id: str
    segment_id: str
    disfluency_type: str
    text: str
    start_time: float
    end_time: float
    duration: float
    audio_clip_path: str
    confidence: float = 1.0


class DisfluencyDetector:
    """Detect disfluencies in Hindi audio transcriptions"""

    # Hindi filler words and patterns
    HINDI_FILLERS = {
        'filler': ['अह', 'अहं', 'उम्म', 'उम', 'एर्म', 'अहा', 'आह', 'हूँ', 'तो', 'न'],
        'hesitation': ['...', '…', 'हाँ', 'जी'],
    }

    # English fillers in Devanagari
    ENGLISH_FILLERS_DEVANAGARI = {
        'filler': ['अह', 'अर्', 'इर्', 'उह', 'ॉह'],
    }

    # Repetition pattern - word repeated 2+ times
    REPETITION_PATTERN = r'(\b\w+\b)(\s+\1)+'

    # Prolongation pattern - repeated characters (sooo, liiiike)
    PROLONGATION_PATTERN = r'(\w)\1{2,}'

    def __init__(self):
        self.disfluencies = []

    def detect_in_text(self, text: str, segment_id: str, recording_id: str,
                       start_time: float, end_time: float) -> List[Disfluency]:
        """Detect disfluencies in text segment"""

        detected = []

        # 1. Check for fillers
        filler_matches = self._detect_fillers(text)
        for filler_text in filler_matches:
            detected.append(Disfluency(
                recording_id=recording_id,
                segment_id=segment_id,
                disfluency_type="filler",
                text=filler_text,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                audio_clip_path="",  # Will be filled later
            ))

        # 2. Check for repetitions
        if self._detect_repetition(text):
            detected.append(Disfluency(
                recording_id=recording_id,
                segment_id=segment_id,
                disfluency_type="repetition",
                text=text,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                audio_clip_path="",
            ))

        # 3. Check for prolongations
        prolongation_words = self._detect_prolongation(text)
        for prolonged_word in prolongation_words:
            detected.append(Disfluency(
                recording_id=recording_id,
                segment_id=segment_id,
                disfluency_type="prolongation",
                text=prolonged_word,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                audio_clip_path="",
            ))

        return detected

    def _detect_fillers(self, text: str) -> List[str]:
        """Detect filler words"""
        fillers = []

        # Check Hindi fillers
        for filler_word in self.HINDI_FILLERS['filler']:
            if filler_word in text:
                fillers.append(filler_word)

        # Check English fillers in Devanagari
        for filler_word in self.ENGLISH_FILLERS_DEVANAGARI['filler']:
            if filler_word in text:
                fillers.append(filler_word)

        return fillers

    def _detect_repetition(self, text: str) -> bool:
        """Detect word repetitions"""
        # Simple pattern: look for repeated words
        words = text.split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                return True
        return False

    def _detect_prolongation(self, text: str) -> List[str]:
        """Detect prolongations (sooo, liiike)"""
        prolonged_words = []
        words = text.split()

        for word in words:
            # Check if word has 3+ repeated consecutive characters
            if re.search(self.PROLONGATION_PATTERN, word):
                prolonged_words.append(word)

        return prolonged_words

    def detect_false_starts(self, text: str) -> bool:
        """Detect false starts - simple heuristic"""
        # Look for patterns like "... started but corrected"
        # This is simplified - you might need more sophisticated logic
        if '...' in text or '…' in text:
            return True
        return False

    def detect_hesitations(self, text: str) -> bool:
        """Detect hesitations"""
        # Long pauses or silence markers
        silence_markers = ['...', '…', '⟨silence⟩', '<silence>']
        return any(marker in text for marker in silence_markers)


# ============================================
# AUDIO SEGMENTATION & CLIPPING
# ============================================

class AudioSegmenter:
    """Segment and clip audio based on timestamps"""

    def __init__(self):
        self.output_dir = ensure_directory(OUTPUTS_DIR / "task2_disfluencies" / "audio_clips")

    def clip_audio(self, audio_path: Path, start_time: float, end_time: float,
                   recording_id: str, segment_id: str) -> Optional[Path]:
        """
        Clip audio segment from full recording

        Args:
            audio_path: Path to full audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            recording_id: Recording ID
            segment_id: Segment ID

        Returns:
            Path to clipped audio file
        """
        try:
            # Load audio
            audio, sr = load_audio(audio_path, sr=SAMPLE_RATE)
            if audio is None:
                logger.error(f"Failed to load audio: {audio_path}")
                return None

            # Segment audio
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Clamp to valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            audio_segment = audio[start_sample:end_sample]

            # Save clipped audio
            output_filename = f"{recording_id}_{segment_id}.wav"
            output_path = self.output_dir / output_filename

            if save_audio(audio_segment, output_path, sr=SAMPLE_RATE):
                return output_path
            else:
                return None

        except Exception as e:
            logger.error(f"Error clipping audio: {str(e)}")
            return None


# ============================================
# MAIN PROCESSING
# ============================================

def download_and_process_data(num_samples: int = None) -> Tuple[List[Dict], List[Dict]]:
    """Download data and extract segments with timestamps"""

    logger.info("Downloading dataset for disfluency detection...")

    # This would be populated from your GCP data
    # For now, we create a template
    data_items = []
    segments = []

    # You'll need to modify this based on actual data structure
    # The transcription JSON should have segment-level information

    # Example structure:
    # {
    #     "recording_id": "123",
    #     "segments": [
    #         {
    #             "segment_id": "seg_1",
    #             "text": "हेलो कैसे हो",
    #             "start_time": 0.5,
    #             "end_time": 2.3
    #         }
    #     ]
    # }

    logger.info(f"Processed {len(data_items)} recordings with {len(segments)} segments")
    return data_items, segments


def main():
    """Main function for Task 2"""
    logger.info("=" * 70)
    logger.info("TASK 2: DISFLUENCY DETECTION & SEGMENTATION")
    logger.info("=" * 70)

    # Setup
    output_dir = ensure_directory(OUTPUTS_DIR / "task2_disfluencies")
    detector = DisfluencyDetector()
    segmenter = AudioSegmenter()

    # Step 1: Download and process data
    logger.info("\n[STEP 1] Loading dataset...")
    recordings, segments = download_and_process_data(
        num_samples=SAMPLE_SIZE if DEBUG else None
    )

    if not segments:
        logger.warning("No segments found. Creating sample data for demonstration...")
        # Create sample data
        segments = [
            {
                "recording_id": "sample_001",
                "segment_id": "seg_1",
                "text": "अह... मैं कहना चाहता हूँ कि यह बहुत अच्छा है",
                "start_time": 0.5,
                "end_time": 3.2,
            },
            {
                "recording_id": "sample_001",
                "segment_id": "seg_2",
                "text": "मैं मैं कहना चाहता हूँ सब कुछ सही है",
                "start_time": 3.5,
                "end_time": 6.1,
            },
            {
                "recording_id": "sample_001",
                "segment_id": "seg_3",
                "text": "यह बहुत सुंदर है...",
                "start_time": 6.5,
                "end_time": 8.2,
            }
        ]

    # Step 2: Detect disfluencies
    logger.info("\n[STEP 2] Detecting disfluencies...")
    all_disfluencies = []

    for segment in get_progress_bar(segments, "Detecting"):
        disfluencies = detector.detect_in_text(
            text=segment["text"],
            segment_id=segment["segment_id"],
            recording_id=segment["recording_id"],
            start_time=segment["start_time"],
            end_time=segment["end_time"]
        )

        all_disfluencies.extend(disfluencies)

    logger.info(f"Detected {len(all_disfluencies)} disfluencies")

    # Step 3: Clip audio (if audio files available)
    logger.info("\n[STEP 3] Clipping audio segments...")

    # This would require actual audio files
    # For now, we just set placeholder paths
    for disfluency in get_progress_bar(all_disfluencies, "Clipping"):
        # In real scenario:
        # disfluency.audio_clip_path = segmenter.clip_audio(...)

        # For demo, create placeholder
        disfluency.audio_clip_path = str(
            segmenter.output_dir / f"{disfluency.recording_id}_{disfluency.segment_id}.wav"
        )

    # Step 4: Create output CSV
    logger.info("\n[STEP 4] Creating output CSV...")

    if all_disfluencies:
        # Convert to DataFrame
        disfluency_dicts = [asdict(d) for d in all_disfluencies]
        df = pd.DataFrame(disfluency_dicts)

        # Save to CSV
        output_csv = output_dir / "disfluencies.csv"
        save_csv(df, output_csv)

        logger.info(f"Saved disfluency CSV with {len(df)} entries")

        # Print summary
        print("\n" + "=" * 70)
        print("DISFLUENCY DETECTION SUMMARY")
        print("=" * 70)
        print(f"\nTotal disfluencies detected: {len(df)}")
        print(f"\nBreakdown by type:")
        print(df['disfluency_type'].value_counts())
        print(f"\nSample disfluencies:")
        print(df.head(10))
        print(f"\nCSV saved: {output_csv}")
    else:
        logger.warning("No disfluencies detected")

    logger.info("\n" + "=" * 70)
    logger.info("TASK 2 COMPLETED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()