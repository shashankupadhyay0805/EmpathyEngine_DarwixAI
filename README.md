# Empathy Engine

A Python service that converts text into **emotionally expressive speech** by detecting the emotion of the text and dynamically modifying voice parameters (pitch, rate, volume) of the text-to-speech output.


## Features

- **Emotion detection**: HuggingFace Transformers (DistilRoBERTa emotion model) with fallback to VADER / TextBlob
- **Voice mapping**: Maps emotions (happy, frustrated, neutral) to pitch, rate, and volume
- **TTS**: gTTS by default; optional ElevenLabs API
- **Audio processing**: pydub for applying pitch/rate/volume
- **API**: FastAPI with `POST /generate-voice`
- **CLI**: One-shot generation and optional server mode
- **Web demo**: Simple HTML page at `/`

## Design choices

- **Emotion detection pipeline**  
  - Primary model: HuggingFace `j-hartmann/emotion-english-distilroberta-base`, which predicts fine‑grained labels such as joy, anger, sadness, fear, disgust, surprise, and neutral.  
  - We map these raw labels to three evaluation‑friendly categories: **happy (Positive/Happy)**, **frustrated (Negative/Frustrated)**, and **neutral** using `EMOTION_TO_CATEGORY` in `config.py`.  
  - If Transformers or GPU support is unavailable, the service transparently falls back to VADER and then TextBlob sentiment so the app still works, just with simpler positive/negative/neutral signals.

- **Emotion → voice parameter mapping**  
  - The mapping lives in `app/voice_mapper.py` via `EMOTION_PARAMS` and `get_speech_params()`.  
  - Each emotion controls **three** vocal parameters:  
    - **Pitch** (tonal height) as a multiplier on the base sample rate (e.g. 1.2 ≈ +20 %),  
    - **Rate** (speech speed) via a playback‑speed‑like frame‑rate change,  
    - **Volume** in decibels (**volume_db**) using pydub gain.  
  - Example mappings:  
    - **happy** → pitch 1.2, rate 1.1, volume_db +2.0 (brighter, faster, louder),  
    - **frustrated** → pitch 0.9, rate 0.85, volume_db −2.0 (flatter, slower, quieter),  
    - **neutral** → pitch 1.0, rate 1.0, volume_db 0.0 (baseline voice).  
  - We also expose an **intensity** parameter (0.0–1.0). Intensity smoothly interpolates these values back toward neutral so the same text can be read slightly or strongly emotional without changing the code.

- **TTS and audio processing**  
  - We generate base speech with **gTTS** (or ElevenLabs if enabled) and then adjust pitch, rate, and volume with **pydub** so the emotion logic is engine‑agnostic.  
  - Outputs are saved to `outputs/generated_audio/` as **.mp3** or **.wav**; both formats are playable and served via `GET /audio/{filename}`.

## Run on your local computer (complete steps)

### 1. Open the project folder

- Open a terminal (Command Prompt, PowerShell, or your IDE terminal).
- Go to the project folder, for example:
  ```bash
  cd C:\Users\shash\Desktop\Empathy
  ```
  (Use your actual path if the project is elsewhere.)

### 2. Create a virtual environment

```bash
python -m venv venv
```

- **Windows (Command Prompt):**  
  `venv\Scripts\activate.bat`
- **Windows (PowerShell):**  
  `.\venv\Scripts\Activate.ps1`
- **macOS/Linux:**  
  `source venv/bin/activate`

You should see `(venv)` at the start of your prompt.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This can take a few minutes (especially `torch` and `transformers` for full emotion detection).

### 4. (Required for audio) FFmpeg

Audio generation needs ffmpeg. Choose one:

- **Option A:** Install [ffmpeg](https://ffmpeg.org/download.html) (e.g. from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)), extract the zip, and add the **bin** folder (the one containing `ffmpeg.exe`) to your system **PATH**.
- **Option B:** Set the **FFMPEG_PATH** environment variable to that **bin** folder, then restart the terminal and the app.
- **Option C (Windows):** Edit `run_server.bat`, set the `FFMPEG_PATH` line to your ffmpeg **bin** path, then start the app with `run_server.bat` (see step 6).

### 5. Start the app (with venv active)

From the project folder, with the virtual environment activated:

```bash
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Or:

```bash
uvicorn app.main:app --reload
```

(If you use `run_server.bat`, it sets `FFMPEG_PATH` and runs the server; skip activating venv manually and run the batch file from the project folder.)

### 6. Open the app

- In your browser go to: **http://localhost:8000**
- Enter text and click **Generate voice**.

To stop the server: press **Ctrl+C** in the terminal.

### If you get "Permission denied" creating the venv

1. **Stop anything using the project**  
   Close any terminal where the server is running (or press Ctrl+C). Close Cursor/VS Code if it’s using the project, then reopen it.

2. **Delete the existing `venv` folder**  
   In File Explorer go to `C:\Users\shash\Desktop\Empathy`, delete the **venv** folder (right‑click → Delete). If it says "in use", restart your PC or run step 1 again.

3. **Create the venv again**  
   Open a **new** terminal, then:
   ```bash
   cd C:\Users\shash\Desktop\Empathy
   python -m venv venv
   ```

4. **If it still says "Permission denied"**  
   - Run the terminal **as Administrator** (right‑click PowerShell/Command Prompt → "Run as administrator"), then `cd` to the project and run `python -m venv venv` again.  
   - Or create the venv in a different folder (e.g. `python -m venv venv2`) and use that instead.

---



### Clone and enter project

```bash
cd Empathy
```

### Create virtual environment


**Note:** For audio processing (MP3 export), [ffmpeg](https://ffmpeg.org/) is required. Either:
- Add the **bin** folder (the one containing `ffmpeg.exe`) to your system PATH, e.g.  
  `C:\...\ffmpeg-8.0.1-essentials_build\bin`, or  
- Set the environment variable **FFMPEG_PATH** to that folder (or to the unpacked parent folder; the app will look for `bin/ffmpeg.exe`), then restart the server.  
- On Windows you can use **run_server.bat**: edit the `FFMPEG_PATH` line to your ffmpeg `bin` path, then double‑click or run it. (Docker image includes ffmpeg.)



### Start API server

```bash
python -m app.main --serve
# Or:
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **Web demo**: Open http://localhost:8000/
- **API**: `POST http://localhost:8000/generate-voice`


## Emotion → voice mapping

| Emotion     | Pitch   | Rate    | Volume  |
|------------|---------|---------|--------|
| **happy**  | +20%    | +10%    | +5%    |
| **frustrated** | −10% | −15%    | −5%    |
| **neutral**| default | default | default |

Implementation uses multipliers (e.g. pitch 1.2, rate 1.1) and volume delta in dB; `intensity` scales the effect toward neutral when &lt; 1.0.

## Project layout

```
project-root/
├── app/
│   ├── main.py           # FastAPI app + CLI
│   ├── emotion_detector.py
│   ├── voice_mapper.py
│   ├── tts_engine.py
│   ├── config.py
│   ├── utils.py
│   └── static/
│       └── demo.html
├── outputs/
│   └── generated_audio/
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
└── README.md
```


