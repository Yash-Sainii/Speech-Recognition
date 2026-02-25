"""
TASK 4: LATTICE-BASED WER COMPUTATION
Build transcription lattices from multiple ASR models
Handle model agreement and fair WER calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import editdistance

from config import *
from utils import *

logger = setup_logger(__name__)


# ============================================
# LATTICE STRUCTURES
# ============================================

@dataclass
class LatticeNode:
    """Node in transcription lattice"""
    position: int
    candidates: List[str]  # Alternative words at this position
    frequency: Dict[str, int]  # Word frequencies

    def get_most_common(self) -> str:
        """Get most common word at this position"""
        if not self.candidates:
            return ""
        return max(self.frequency.items(), key=lambda x: x[1])[0]

    def get_confidence(self, word: str) -> float:
        """Get confidence for specific word"""
        if word not in self.frequency:
            return 0.0
        return self.frequency[word] / sum(self.frequency.values())


class TranscriptionLattice:
    """Lattice representing all transcription alternatives"""

    def __init__(self, num_positions: int):
        self.num_positions = num_positions
        self.nodes = {}  # position -> LatticeNode

    def add_transcription(self, transcription: List[str]):
        """Add a transcription to the lattice"""
        for pos, word in enumerate(transcription):
            if pos not in self.nodes:
                self.nodes[pos] = LatticeNode(
                    position=pos,
                    candidates=[],
                    frequency=Counter()
                )

            node = self.nodes[pos]
            if word not in node.candidates:
                node.candidates.append(word)
            node.frequency[word] += 1

    def to_dict(self) -> Dict:
        """Convert lattice to dictionary representation"""
        return {
            pos: {
                'candidates': list(node.candidates),
                'frequency': dict(node.frequency),
                'most_common': node.get_most_common()
            }
            for pos, node in self.nodes.items()
        }

    def find_best_path(self, reference: List[str],
                       agreement_threshold: float = 0.6) -> List[str]:
        """
        Find best path through lattice considering reference and model agreement

        Args:
            reference: Reference transcription
            agreement_threshold: Threshold for model agreement (e.g., 0.6 = 60%)

        Returns:
            Best path through lattice
        """
        best_path = []

        for pos in range(self.num_positions):
            if pos not in self.nodes:
                # No data for this position, use reference if available
                if pos < len(reference):
                    best_path.append(reference[pos])
                continue

            node = self.nodes[pos]

            # Get model agreement
            total_models = sum(node.frequency.values())
            most_common_word = node.get_most_common()
            most_common_freq = node.frequency[most_common_word]
            agreement_level = most_common_freq / total_models if total_models > 0 else 0

            # Decision logic
            if agreement_level >= agreement_threshold:
                # Models agree, use majority vote
                best_path.append(most_common_word)
            elif pos < len(reference) and reference[pos] in node.candidates:
                # Reference agrees with a model, use reference
                best_path.append(reference[pos])
            else:
                # No clear consensus, use model majority
                best_path.append(most_common_word)

        return best_path

    def visualize(self, max_positions: int = 10) -> str:
        """Visualize lattice as ASCII art"""
        viz = "Lattice Visualization:\n"

        for pos in sorted(self.nodes.keys())[:max_positions]:
            node = self.nodes[pos]
            candidates_str = " | ".join(f"{w}({node.frequency[w]})"
                                        for w in node.candidates)
            viz += f"Pos {pos}: [{candidates_str}]\n"

        if self.num_positions > max_positions:
            viz += f"... (and {self.num_positions - max_positions} more positions)\n"

        return viz


# ============================================
# WER COMPUTATION
# ============================================

class WERCalculator:
    """Calculate WER with various metrics"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove extra spaces and split
        return text.strip().split()

    @staticmethod
    def compute_wer(reference: List[str], hypothesis: List[str]) -> float:
        """
        Compute Word Error Rate

        WER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = reference length
        """

        if len(reference) == 0:
            return 0.0 if len(hypothesis) == 0 else float('inf')

        # Use dynamic programming (Levenshtein distance)
        d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)

        for i in range(len(reference) + 1):
            d[i][0] = i
        for j in range(len(hypothesis) + 1):
            d[0][j] = j

        for i in range(1, len(reference) + 1):
            for j in range(1, len(hypothesis) + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return float(d[len(reference)][len(hypothesis)]) / len(reference)

    @staticmethod
    def compute_metrics(reference: List[str], hypothesis: List[str]) -> Dict:
        """Compute detailed WER metrics"""

        wer = WERCalculator.compute_wer(reference, hypothesis)

        # Detailed breakdown
        ref_len = len(reference)
        hyp_len = len(hypothesis)

        # Get alignment
        d = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)
        for i in range(ref_len + 1):
            d[i][0] = i
        for j in range(hyp_len + 1):
            d[0][j] = j

        # Edit distance
        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j])

        # Count operations
        i, j = ref_len, hyp_len
        substitutions = 0
        deletions = 0
        insertions = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i - 1][j - 1] < d[i - 1][j] and d[i - 1][j - 1] < d[i][j - 1]:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i][j - 1] < d[i - 1][j]:
                insertions += 1
                j -= 1
            else:
                deletions += 1
                i -= 1

        return {
            'wer': wer,
            'reference_length': ref_len,
            'hypothesis_length': hyp_len,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
        }


# ============================================
# MODEL EVALUATION
# ============================================

class ModelEvaluator:
    """Evaluate ASR models using lattice-based WER"""

    def __init__(self):
        self.wer_calc = WERCalculator()

    def evaluate_models(self, test_cases: List[Dict],
                        agreement_threshold: float = 0.6) -> pd.DataFrame:
        """
        Evaluate multiple models on test cases

        Args:
            test_cases: List of dicts with:
                - reference: Reference transcription
                - model1, model2, ...: Model outputs
            agreement_threshold: Model agreement threshold

        Returns:
            DataFrame with results
        """

        results = []

        for test_case in get_progress_bar(test_cases, "Evaluating"):

            # Extract reference
            reference_text = test_case.get('reference', '')
            reference = self.wer_calc.tokenize(reference_text)

            # Build lattice from model outputs
            model_names = [k for k in test_case.keys() if k.startswith('model')]
            transcriptions = [self.wer_calc.tokenize(test_case[name])
                              for name in model_names]

            # Find max length for lattice
            max_len = max(len(t) for t in transcriptions) if transcriptions else 0

            # Build lattice
            lattice = TranscriptionLattice(max_len)
            for trans in transcriptions:
                lattice.add_transcription(trans)

            # Find best path
            best_path = lattice.find_best_path(reference, agreement_threshold)

            # Evaluate each model
            for model_name, transcription in zip(model_names, transcriptions):
                # Original WER (with reference)
                original_metrics = self.wer_calc.compute_metrics(reference, transcription)

                # Lattice WER (with best path)
                lattice_metrics = self.wer_calc.compute_metrics(best_path, transcription)

                improvement = original_metrics['wer'] - lattice_metrics['wer']
                improvement_pct = (improvement / original_metrics['wer'] * 100) if original_metrics['wer'] > 0 else 0

                results.append({
                    'test_case_id': test_case.get('id', 'unknown'),
                    'model_name': model_name,
                    'reference_text': reference_text,
                    'hypothesis_text': test_case[model_name],
                    'original_wer': round(original_metrics['wer'], 4),
                    'lattice_wer': round(lattice_metrics['wer'], 4),
                    'wer_improvement': round(improvement, 4),
                    'improvement_percentage': round(improvement_pct, 2),
                    'reference_length': original_metrics['reference_length'],
                    'substitutions': original_metrics['substitutions'],
                    'deletions': original_metrics['deletions'],
                    'insertions': original_metrics['insertions'],
                })

        return pd.DataFrame(results)


# ============================================
# MAIN PROCESSING
# ============================================

def generate_sample_test_cases() -> List[Dict]:
    """Generate sample test cases for demonstration"""

    test_cases = [
        {
            'id': 'test_001',
            'reference': 'नमस्ते मेरा नाम राज है',
            'model1': 'नमस्ते मेरा नाम राज है',  # Perfect
            'model2': 'नमस्ते मेरा नाम राजेश है',  # Substitution
            'model3': 'नमस्ते मेरा नाम है',  # Deletion
            'model4': 'नमस्ते मेरा नाम राज कुमार है',  # Insertion
            'model5': 'नमस्ते का नाम राज है',  # Substitution
        },
        {
            'id': 'test_002',
            'reference': 'क्या आप कैसे हो',
            'model1': 'क्या आप कैसे हो',  # Perfect
            'model2': 'क्या आप ठीक हो',  # Substitution
            'model3': 'क्या कैसे हो',  # Deletion
            'model4': 'क्या आप वाकई कैसे हो',  # Insertion
            'model5': 'क्या आप कहाँ हो',  # Substitution
        },
        {
            'id': 'test_003',
            'reference': 'यह एक अच्छा दिन है',
            'model1': 'यह एक अच्छा दिन है',  # Perfect
            'model2': 'यह अच्छा दिन है',  # Deletion
            'model3': 'यह एक बहुत अच्छा दिन है',  # Insertion
            'model4': 'यह एक बुरा दिन है',  # Substitution
            'model5': 'यह दिन अच्छा है',  # Word order
        },
    ]

    return test_cases


def main():
    """Main function for Task 4"""
    logger.info("=" * 70)
    logger.info("TASK 4: LATTICE-BASED WER COMPUTATION")
    logger.info("=" * 70)

    # Setup
    output_dir = ensure_directory(OUTPUTS_DIR / "task4_lattice_wer")
    evaluator = ModelEvaluator()

    # Step 1: Load test cases
    logger.info("\n[STEP 1] Loading test cases...")

    # Try to load from file
    test_cases_file = RAW_DATA_DIR / "test_cases.json"
    if test_cases_file.exists():
        try:
            with open(test_cases_file, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            logger.info(f"Loaded {len(test_cases)} test cases from file")
        except Exception as e:
            logger.warning(f"Could not load test cases file: {e}")
            test_cases = generate_sample_test_cases()
    else:
        test_cases = generate_sample_test_cases()
        logger.info(f"Using {len(test_cases)} sample test cases")

    # Step 2: Evaluate models with lattice
    logger.info("\n[STEP 2] Evaluating models with lattice-based WER...")

    df_results = evaluator.evaluate_models(
        test_cases,
        agreement_threshold=0.6
    )

    # Step 3: Save results
    logger.info("\n[STEP 3] Saving results...")

    output_csv = output_dir / "lattice_wer_results.csv"
    save_csv(df_results, output_csv)

    # Step 4: Generate summary and visualization
    logger.info("\n[STEP 4] Generating summary...")

    print("\n" + "=" * 70)
    print("LATTICE-BASED WER EVALUATION SUMMARY")
    print("=" * 70)

    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"Total test cases: {df_results['test_case_id'].nunique()}")
    print(f"Total models evaluated: {df_results['model_name'].nunique()}")
    print(f"Total evaluations: {len(df_results)}")

    # WER comparison by model
    print(f"\n📈 WER Comparison by Model:")
    model_summary = df_results.groupby('model_name').agg({
        'original_wer': ['mean', 'std'],
        'lattice_wer': ['mean', 'std'],
        'wer_improvement': 'mean',
        'improvement_percentage': 'mean'
    }).round(4)

    print(model_summary)

    # Models that benefited from lattice
    print(f"\n✨ Models with Improvement (WER reduced):")
    improved = df_results[df_results['wer_improvement'] > 0].groupby('model_name').agg({
        'wer_improvement': 'mean',
        'improvement_percentage': 'mean'
    }).sort_values('improvement_percentage', ascending=False)

    for model, row in improved.iterrows():
        print(f"  {model:15} - Avg improvement: {row['wer_improvement']:.4f} ({row['improvement_percentage']:.1f}%)")

    # Sample results
    print(f"\n📋 Sample Results (first 5):")
    sample_cols = ['test_case_id', 'model_name', 'original_wer', 'lattice_wer', 'improvement_percentage']
    print(df_results[sample_cols].head(10).to_string(index=False))

    # Save detailed results
    detailed_output = output_dir / "detailed_lattice_results.json"
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(df_results.to_dict('records'), f, ensure_ascii=False, indent=2)

    # Save summary statistics
    summary_stats = {
        'total_test_cases': int(df_results['test_case_id'].nunique()),
        'total_models': int(df_results['model_name'].nunique()),
        'avg_original_wer': float(df_results['original_wer'].mean()),
        'avg_lattice_wer': float(df_results['lattice_wer'].mean()),
        'avg_improvement_percentage': float(df_results['improvement_percentage'].mean()),
        'models_improved': int(len(improved)),
    }

    summary_file = output_dir / "summary_stats.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults saved to: {output_csv}")
    logger.info(f"Detailed results saved to: {detailed_output}")
    logger.info(f"Summary stats saved to: {summary_file}")

    logger.info("\n" + "=" * 70)
    logger.info("TASK 4 COMPLETED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()