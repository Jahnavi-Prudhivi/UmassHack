import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from elevenlabs.client import ElevenLabs

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")  # <- set this in .env

def _fail(msg: str):
    print(f"❌ {msg}")
    raise SystemExit(1)

if not GEMINI_KEY:
    _fail("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
if not ELEVEN_KEY:
    _fail("Missing ELEVENLABS_API_KEY in .env")

genai.configure(api_key=GEMINI_KEY)
client = ElevenLabs(api_key=ELEVEN_KEY)

PREFERRED_ORDER = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-pro-latest",
]

def pick_model_name() -> str:
    try:
        models = list(genai.list_models())
    except Exception as e:
        _fail(f"Could not list models: {e}")

    supports = {}
    for m in models:
        methods = set(getattr(m, "supported_generation_methods", []))
        supports[m.name] = ("generateContent" in methods)

    for prefer in PREFERRED_ORDER:
        for name in supports:
            if name.endswith(prefer) and supports[name]:
                return name

    for name in supports:
        if "gemini" in name and supports[name]:
            return name

    _fail("No Gemini text model with 'generateContent' is available for this API key.")

MODEL_NAME = pick_model_name()
print(f"Using Gemini model: {MODEL_NAME}")

def correct_text(text: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            "You convert noisy ASL fingerspelling (letters only) into clean English. "
            "Fix repeated letters, infer word boundaries, and add proper grammar & punctuation. "
            "Output ONLY the final corrected sentence.\n\n"
            f"RAW: {text}"
        )
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        _fail(f"Gemini request failed: {e}")

def speak_text(text: str, output_file: str = "asl_audio.mp3"):
    """
    Uses a provided voice_id (no 'voices_read' permission required).
    """
    voice_id = ELEVEN_VOICE_ID or "21m00Tcm4TlvDq8ikWAM"  # example “Rachel”; replace if your account differs
    print(f"🔊 Generating speech with voice_id={voice_id}...")
    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
        )
        data = b"".join(chunk for chunk in audio)
        Path(output_file).write_bytes(data)
        print(f"Saved → {output_file}")
    except Exception as e:
        _fail(f"ElevenLabs TTS failed: {e}")

if __name__ == "__main__":
    letters_path = Path(r"..\ASL-Gesture-App-Hack\letters.csv")
    if not letters_path.exists():
        _fail(f"letters.csv not found at {letters_path.resolve()}")

    with letters_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if lines and lines[0].lower().startswith("letter"):
        lines = lines[1:]

    raw_text = "".join(lines)
    print(f"Raw letters: {raw_text}")

    corrected_text = correct_text(raw_text)
    print(f"Gemini corrected text: {corrected_text}")

    speak_text(corrected_text, output_file="asl_audio.mp3")
