# asl_run_auto.py
import subprocess
from pathlib import Path
import time
import os
from joblib import load

# ---------------- Absolute Paths ----------------
LETTERS_CSV = Path(r"C:\Users\Janu\Desktop\hack\ASL-Gesture-App-Debug\ASL-Gesture-App-Hack\letters.csv")
BRIDGE_SCRIPT = Path(r"C:\Users\Janu\Desktop\hack\ASL-Gesture-App-Debug\ASL-Gesture-App-NLP\bridge_nlp.py")
ASL_INFERENCE_SCRIPT = Path(r"C:\Users\Janu\Desktop\hack\ASL-Gesture-App-Debug\ASL-Gesture-App-Hack\inference.py")
AUDIO_FILE = Path(r"C:\Users\Janu\Desktop\hack\ASL-Gesture-App-Debug\ASL-Gesture-App-NLP\asl_audio.mp3")
MODEL_PATH = r"C:\Users\Janu\Desktop\hack\ASL-Gesture-App-Debug\ASL-Gesture-App-Hack\models\asl_model.joblib"

# ---------------- Functions ----------------
def run_in_conda(env: str, script_path: Path):
    """Run a Python script in a conda environment and print output"""
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return

    print(f"🔹 Running {script_path.name} in conda env '{env}'...")
    proc = subprocess.Popen(
        ["conda", "run", "-n", env, "python", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in proc.stdout:
        print(line.strip())
    proc.wait()
    if proc.returncode != 0:
        print(f"⚠️ {script_path.name} exited with code {proc.returncode}")
    else:
        print(f"✅ {script_path.name} finished successfully")

def read_letters_csv() -> str:
    """Read letters.csv and return continuous string"""
    if LETTERS_CSV.exists():
        with LETTERS_CSV.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return "".join(lines)
    return ""

# ---------------- Main Workflow ----------------
if __name__ == "__main__":
    input("Press Enter to start ASL workflow (camera will open)...")

    # 1️⃣ Run ASL inference
    run_in_conda("asl311", ASL_INFERENCE_SCRIPT)

    # 2️⃣ Read letters
    letters = read_letters_csv()
    if not letters:
        print("⚠️ No letters detected in letters.csv")
    else:
        print(f"📝 Letters detected: {letters}")

        # 3️⃣ Run NLP + TTS
        run_in_conda("asl-nlp", BRIDGE_SCRIPT)
        if AUDIO_FILE.exists():
            print(f"🎵 Audio saved: {AUDIO_FILE}")
        else:
            print("⚠️ Audio file not generated")
