"""
TASK 3: SPELLING ERROR DETECTION & CORRECTION
Identify correct vs incorrect spelling in Hindi words
Handle English words transcribed in Devanagari script
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging
import json
import re
from collections import Counter, defaultdict

from config import *
from utils import *

logger = setup_logger(__name__)


# ============================================
# DEVANAGARI & HINDI VALIDATION
# ============================================

class DevanagariValidator:
    """Validate Devanagari script and Hindi spelling"""

    # Valid Devanagari Unicode ranges
    DEVANAGARI_RANGE = (0x0900, 0x097F)

    # Common Devanagari characters
    VOWELS = ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ']

    CONSONANTS = [
        'क', 'ख', 'ग', 'घ', 'ङ',
        'च', 'छ', 'ज', 'झ', 'ञ',
        'ट', 'ठ', 'ड', 'ढ', 'ण',
        'त', 'थ', 'द', 'ध', 'न',
        'प', 'फ', 'ब', 'भ', 'म',
        'य', 'र', 'ल', 'ळ',
        'व', 'श', 'ष', 'स', 'ह'
    ]

    # Diacritics (vowel signs)
    DIACRITICS = [
        '\u093E',  # ा (aa)
        '\u093F',  # ि (i)
        '\u0940',  # ी (ii)
        '\u0941',  # ु (u)
        '\u0942',  # ू (uu)
        '\u0943',  # ृ (ri)
        '\u0944',  # ॄ (rii)
        '\u0945',  # ॅ (e - candra e)
        '\u0946',  # ॆ (e)
        '\u0947',  # े (e)
        '\u0948',  # ै (ai)
        '\u0949',  # ॉ (o - candra o)
        '\u094A',  # ॊ (o)
        '\u094B',  # ो (o)
        '\u094C',  # ौ (au)
    ]

    # Common Hindi words dictionary (sample - you should expand this)
    HINDI_DICTIONARY = {
        'नमस्ते', 'धन्यवाद', 'कृपया', 'मैं', 'तुम', 'वह', 'यह', 'हम',
        'घर', 'गाड़ी', 'पेड़', 'पानी', 'खाना', 'पीना', 'सोना', 'जागना',
        'जाना', 'आना', 'देना', 'लेना', 'करना', 'होना', 'कहना', 'सुनना',
        'देखना', 'मिलना', 'रहना', 'बैठना', 'खड़ा', 'चलना', 'दौड़ना',
        'मां', 'बाप', 'भाई', 'बहन', 'बेटा', 'बेटी', 'दोस्त', 'शिक्षक',
        'अच्छा', 'बुरा', 'छोटा', 'बड़ा', 'लंबा', 'छोटी', 'सुंदर', 'कुरूप',
        'सफ़ेद', 'काला', 'लाल', 'पीला', 'नीला', 'हरा', 'गुलाबी', 'नारंगी',
    }

    def __init__(self):
        self.valid_chars = set(self.VOWELS + self.CONSONANTS + self.DIACRITICS)

    def is_valid_devanagari(self, word: str) -> bool:
        """Check if word contains valid Devanagari characters"""
        if not word:
            return False

        for char in word:
            char_code = ord(char)
            # Allow Devanagari range and spaces
            if not ((self.DEVANAGARI_RANGE[0] <= char_code <= self.DEVANAGARI_RANGE[1]) or
                    char.isspace()):
                return False

        return True

    def has_devanagari(self, word: str) -> bool:
        """Check if word contains any Devanagari character"""
        for char in word:
            if self.DEVANAGARI_RANGE[0] <= ord(char) <= self.DEVANAGARI_RANGE[1]:
                return True
        return False

    def is_likely_english_word(self, word: str) -> bool:
        """Check if word is likely an English word transcribed in Devanagari"""
        # Count ASCII letters
        ascii_count = sum(1 for c in word if ord(c) < 128)
        # If word is mostly ASCII, might be English
        return ascii_count > 0

    def has_character_errors(self, word: str) -> bool:
        """Check for common character errors/typos"""

        if not self.is_valid_devanagari(word):
            return True

        # Check for doubled/tripled consonants without virama
        consonant_str = ''.join(self.CONSONANTS)
        for i in range(len(word) - 2):
            if word[i] in consonant_str and word[i] == word[i + 1] == word[i + 2]:
                return True  # Repeated consonant 3+ times likely error

        return False

    def is_known_word(self, word: str) -> bool:
        """Check if word is in known Hindi dictionary"""
        return word in self.HINDI_DICTIONARY

    def classify_spelling(self, word: str) -> Tuple[str, str]:
        """
        Classify word as correct or incorrect spelling

        Returns:
            Tuple of (status, reason)
        """

        # Empty word
        if not word or not word.strip():
            return "incorrect", "Empty word"

        # Check if valid Devanagari
        if not self.is_valid_devanagari(word):
            return "incorrect", "Invalid Devanagari characters"

        # Check for character errors
        if self.has_character_errors(word):
            return "incorrect", "Character errors/typos"

        # Check if known Hindi word
        if self.is_known_word(word):
            return "correct", "Known Hindi word"

        # English word in Devanagari is acceptable
        # (e.g., कंप्यटू र for "computer")
        if self.is_likely_english_word(word):
            # Check if transcription is reasonable
            if len(word) > 2 and self.is_valid_devanagari(word):
                return "correct", "English word in Devanagari"

        # If made of valid characters but not in dictionary
        # Might be proper noun, rare word, or dialect
        return "correct", "Valid structure (proper noun or rare word)"


# ============================================
# SPELLING CORRECTION
# ============================================

class SpellingCorrector:
    """Detect and classify spelling errors"""

    def __init__(self):
        self.validator = DevanagariValidator()
        self.classification_results = []

    def classify_words(self, words: List[str]) -> pd.DataFrame:
        """Classify list of words"""

        results = []

        for word in get_progress_bar(words, "Classifying words"):
            status, reason = self.validator.classify_spelling(word)

            results.append({
                'word': word,
                'spelling_status': status,
                'reason': reason,
                'is_devanagari': self.validator.has_devanagari(word),
                'length': len(word),
            })

        return pd.DataFrame(results)

    def load_word_list(self, file_path: Path) -> List[str]:
        """Load word list from file (JSON, CSV, or text)"""

        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'words' in data:
                        return data['words']

            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
                # Assume first column has words
                return df.iloc[:, 0].tolist()

            else:  # Text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]

        except Exception as e:
            logger.error(f"Error loading word list: {str(e)}")
            return []


# ============================================
# MAIN PROCESSING
# ============================================

def load_sample_words() -> List[str]:
    """Load sample words for demonstration"""

    # Mix of correct and incorrect words
    sample_words = [
        # Correct Hindi words
        'नमस्ते',
        'धन्यवाद',
        'घर',
        'पानी',
        'खाना',

        # English words in Devanagari (correct)
        'कंप्यटू र',  # computer
        'फोन',  # phone
        'ईमेल',  # email
        'इंटरनेट',  # internet

        # Incorrect/typos
        'नमस्तेे',  # Double character
        'घरर',  # Typo
        'खान',  # Incomplete
        'पानीी',  # Double character
        'धन्यवादद',  # Double character

        # More examples
        'शिक्षक',  # teacher (correct)
        'छात्र',  # student (correct)
        'पुस्तक',  # book (correct)
        'स्कूल',  # school (correct)
        'मैं',  # I (correct)
        'आप',  # You (correct)
        'वह',  # He/She (correct)

        # Questionable/rare
        'अनिर्भर',  # rare/proper
        'स्वतंत्रता',  # correct (freedom)
    ]

    return sample_words


def main():
    """Main function for Task 3"""
    logger.info("=" * 70)
    logger.info("TASK 3: SPELLING ERROR DETECTION")
    logger.info("=" * 70)

    # Setup
    output_dir = ensure_directory(OUTPUTS_DIR / "task3_spelling")
    corrector = SpellingCorrector()

    # Step 1: Load words
    logger.info("\n[STEP 1] Loading word list...")

    # Try to load from file if it exists
    word_list_path = RAW_DATA_DIR / "hindi_words.json"
    if word_list_path.exists():
        words = corrector.load_word_list(word_list_path)
        logger.info(f"Loaded {len(words)} words from file")
    else:
        # Use sample words for demonstration
        words = load_sample_words()
        logger.info(f"Using {len(words)} sample words for demonstration")

    # Remove duplicates
    words = list(set(words))
    logger.info(f"Processing {len(words)} unique words")

    # Step 2: Classify words
    logger.info("\n[STEP 2] Classifying spelling...")
    df_results = corrector.classify_words(words)

    # Step 3: Save results
    logger.info("\n[STEP 3] Saving results...")

    output_csv = output_dir / "spelling_classification.csv"
    save_csv(df_results, output_csv)

    # Step 4: Generate summary
    logger.info("\n[STEP 4] Generating summary...")

    correct_count = len(df_results[df_results['spelling_status'] == 'correct'])
    incorrect_count = len(df_results[df_results['spelling_status'] == 'incorrect'])

    print("\n" + "=" * 70)
    print("SPELLING CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal unique words analyzed: {len(df_results)}")
    print(f"Correct spelling: {correct_count}")
    print(f"Incorrect spelling: {incorrect_count}")
    print(f"Percentage correct: {100 * correct_count / len(df_results):.1f}%")

    print(f"\n\nClassification Breakdown:")
    print(df_results['spelling_status'].value_counts())

    print(f"\n\nReason Breakdown (top 10):")
    print(df_results['reason'].value_counts().head(10))

    print(f"\n\nSample Correct Words (10):")
    correct_words = df_results[df_results['spelling_status'] == 'correct'].head(10)
    for idx, row in correct_words.iterrows():
        print(f"  {row['word']:20} - {row['reason']}")

    print(f"\n\nSample Incorrect Words (10):")
    incorrect_words = df_results[df_results['spelling_status'] == 'incorrect'].head(10)
    for idx, row in incorrect_words.iterrows():
        print(f"  {row['word']:20} - {row['reason']}")

    # Save summary statistics
    summary_stats = {
        'total_words': len(df_results),
        'correct_count': int(correct_count),
        'incorrect_count': int(incorrect_count),
        'correct_percentage': float(100 * correct_count / len(df_results))
    }

    summary_file = output_dir / "summary_stats.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults saved to: {output_csv}")
    logger.info(f"Summary stats saved to: {summary_file}")

    logger.info("\n" + "=" * 70)
    logger.info("TASK 3 COMPLETED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()