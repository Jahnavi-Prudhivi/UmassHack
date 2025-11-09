# ASL-Gesture-App-Hack

## Overview
This project aims to **bridge communication between the Deaf/Hard-of-Hearing and hearing communities** by translating **American Sign Language (ASL)** gestures into **spoken English** using AI-driven language and speech models.

Our system:
1. **Recognizes ASL gestures in real time** via a webcam.
2. **Translates gestures into English text** using machine learning (TensorFlow/Mediapipe).
3. **Refines and expands text** using **Google Gemini API** for fluent language output.
4. **Generates expressive speech** using **ElevenLabs API**.

---
## Demo Video Link: 
https://drive.google.com/file/d/1z5NFGEtjW_7V4KNjXR1rBofghgYUgjty/view?usp=sharing
---



## Key Components
| Module | Function | Tools/Libraries |
|--------|-----------|----------------|
| **Gesture Recognition** | Detects and classifies ASL hand signs | Mediapipe, OpenCV, TensorFlow Lite |
| **Language Refinement** | Converts literal ASL text → natural English | Google Gemini API |
| **Voice Synthesis** | Converts refined text → natural voice | ElevenLabs Text-to-Speech API |
| **User Interface** | Displays live video, detected sign, and translated output | Streamlit (or Flask) |
| **Optional Deployment** | Portable device version | Raspberry Pi 5 or 4 with Pi Camera |

---

## Project Structure


ASL-Gesture-Application/
│
├── src/
│   ├── gesture_recognition.py     # Hand tracking + gesture classification
│   ├── gemini_refinement.py       # Gemini API text refinement
│   ├── tts_elevenlabs.py          # Text-to-speech output
│   ├── main.py                    # Main entrypoint (combines all modules)
│
├── models/
│   ├── asl_model.tflite           # Pretrained ASL gesture model
│
├── assets/
│   ├── demo_video.mp4             # Optional demo recording
│
├── requirements.txt
├── .env.example                   # Example API key file
├── README.md
└── LICENSE



---

## Setup Instructions

### 1️ Clone the Repository
```bash
git clone https://github.com/Jahnavi-Prudhivi/ASL-Gesture-Application.git
cd ASL-Gesture-Application
````

---

### 2️Create a Virtual Environment

**Mac / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt` yet, create one with:

```txt
mediapipe
opencv-python
tensorflow
requests
streamlit
playsound
python-dotenv
```

---

### 4️ API Key Setup

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following lines:

```bash
GEMINI_API_KEY=your_google_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

---

### 5️ Run the Application

```bash
python src/main.py
```

or, if using Streamlit:

```bash
streamlit run src/main.py
```

You should see a live webcam feed, detected gestures, text translation, and hear voice output.

---

## Optional: Raspberry Pi Setup

If you want to deploy it on a **Raspberry Pi (Pi 4/5)** for a portable demo:

###  Hardware Needed

* Raspberry Pi 4 or 5 (8GB preferred)
* Pi Camera (CSI or USB)
* Small speaker (3.5mm jack or Bluetooth)
* 5" LCD (optional, for portable display)

###  Setup Commands

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-opencv portaudio19-dev -y
pip3 install mediapipe tensorflow-lite requests playsound python-dotenv
```

Then run:

```bash
python3 src/main.py
```

---

##  Development Workflow

1. **Create a new branch** for your work:

   ```bash
   git checkout -b feature-gesture-detection
   ```
2. **Make changes and commit**:

   ```bash
   git add .
   git commit -m "Add ASL hand detection"
   ```
3. **Push your branch**:

   ```bash
   git push origin feature-gesture-detection
   ```
4. **Create a Pull Request** on GitHub to merge your changes into `main`.

---

## 💡 Suggested Roles (Hackathon)

| Role               | Team Member | Focus Area                                |
| ------------------ | ----------- | ----------------------------------------- |
| ML/Computer Vision | Vivek       | Gesture detection & Mediapipe integration |
| AI Integration     | Teammate 2  | Gemini API and NLP flow                   |
| Frontend/Audio     | Teammate 3  | Streamlit UI + ElevenLabs output          |

---

## Vision

A real-time **ASL-to-Speech Translator** that empowers inclusive communication — blending **AI language models, computer vision, and expressive voice synthesis** to help the world understand sign language more naturally.

---

## Stretch Goals

* Detect full ASL **words/phrases** (not just letters)
* Add **voice-to-sign** reverse translation
* Deploy as a **mobile or wearable** app
* Integrate with **smart glasses or AR**

---

## Hackathon Pitch Summary

> “We built a real-time ASL Translator that uses computer vision to read signs, Gemini AI to understand language context, and ElevenLabs to speak them aloud — creating instant, inclusive communication.”

---

## References

* [Mediapipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
* [TensorFlow Lite Models](https://www.tensorflow.org/lite/models)
* [Google Gemini API](https://ai.google.dev/)
* [ElevenLabs API](https://api.elevenlabs.io/)

---

## Contributors

* **Jahnavi Prudhivi** – Computer Vision & Integration
* **Vivek Reddy Kasireddy** – API Integration & Voice Output
* **Aishwarya Silam** – UI/UX & System Architecture

---

## License

This project is open-source under the [MIT License](LICENSE).
