"""
MAIN.PY - ENTRY POINT FOR ALL TASKS
Run this file to execute all 4 tasks sequentially
"""

import os
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import logger, DEBUG, SAMPLE_SIZE

def print_header(task_name: str, task_number: int):
    """Print task header"""
    print("\n" + "="*70)
    print(f"  TASK {task_number}: {task_name}")
    print("="*70 + "\n")

def main():
    """Main function - run all tasks"""

    print("\n" + "🔷"*35)
    print("  HINDI ASR RESEARCH PROJECT - ALL TASKS")
    print("🔷"*35 + "\n")

    logger.info("Starting Hindi ASR Research Project")
    logger.info(f"Debug Mode: {DEBUG}")
    if SAMPLE_SIZE:
        logger.info(f"Sample Size: {SAMPLE_SIZE}")

    # Task 1
    try:
        print_header("Whisper Fine-tuning for Hindi ASR", 1)
        # Try to import updated version first
        try:
            from task1_whisper_finetuning_v2 import main as task1_main
        except ImportError:
            from task1_whisper_finetuning import main as task1_main
        task1_main()
        print("\n✅ Task 1 completed successfully")
    except Exception as e:
        print(f"\n⚠️  Task 1 skipped: {str(e)}")
        logger.warning(f"Task 1 skipped: {str(e)}")

    # Task 2
    try:
        print_header("Disfluency Detection & Segmentation", 2)
        from task2_disfluency_detection import main as task2_main
        task2_main()
        print("\n✅ Task 2 completed successfully")
    except Exception as e:
        print(f"\n❌ Task 2 failed: {str(e)}")
        logger.error(f"Task 2 failed: {str(e)}")

    # Task 3
    try:
        print_header("Spelling Error Correction", 3)
        from task3_spelling_correction import main as task3_main
        task3_main()
        print("\n✅ Task 3 completed successfully")
    except Exception as e:
        print(f"\n❌ Task 3 failed: {str(e)}")
        logger.error(f"Task 3 failed: {str(e)}")

    # Task 4
    try:
        print_header("Lattice-based WER Computation", 4)
        from task4_lattice_wer import main as task4_main
        task4_main()
        print("\n✅ Task 4 completed successfully")
    except Exception as e:
        print(f"\n❌ Task 4 failed: {str(e)}")
        logger.error(f"Task 4 failed: {str(e)}")

    # Final summary
    print("\n" + "="*70)
    print("  PROJECT COMPLETE!")
    print("="*70)
    print("\nCheck 'outputs/' folder for results!")
    logger.info("Project execution completed")

if __name__ == "__main__":
    main()