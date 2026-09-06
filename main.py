import subprocess
import sys


def run_step(description, script_path):
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script_path],
        check=False
    )

    if result.returncode != 0:
        print(f"\n❌ Failed: {script_path}")
        sys.exit(result.returncode)

    print(f"✅ Completed: {script_path}")


def main():
    print("=" * 70)
    print("OPTIMIZED SPAM DETECTION SYSTEM")
    print("End-to-End Machine Learning Pipeline")
    print("=" * 70)

    run_step(
        "STEP 1 — DATA PREPROCESSING",
        "src/preprocess.py"
    )

    run_step(
        "STEP 2 — FEATURE ENGINEERING & MODEL TRAINING",
        "src/train.py"
    )

    run_step(
        "STEP 3 — MODEL EVALUATION & TUNING",
        "src/evaluate.py"
    )

    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nTo launch the application:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()
    
