# 🎵 Harmoniq - Voice & AI Music Composer

**Hum a melody or click notes, then let AI complete your song!**

## Features

- 🎤 **Voice Input**: Record yourself humming/singing - AI converts to musical notes
- 🎹 **Visual Music Staff**: Click to add notes manually
- 🤖 **AI Completion**: OpenAI or Gemini completes your melody based on your prompt
- ▶️ **Instant Playback**: Hear your music immediately
- 💾 **Export**: Download your creation (coming soon)

## Quick Start

1. **Install dependencies:**
```bash
cd Harmoniq
pip install -r requirements.txt
```

2. **Run the app:**
```bash
python app.py
```

3. **Open browser:**
```
http://localhost:5000
```

4. **Get started:**
   - Enter your OpenAI or Gemini API key
   - Either:
     - Record yourself humming a melody (click "Start Recording")
     - OR click on the piano keys to add notes manually
   - Write what you want in the prompt box (e.g., "make it jazzy", "add a sad ending")
   - Click "Generate with AI"
   - Listen to your AI-completed song!

## Requirements

- Python 3.8+
- OpenAI API key OR Google Gemini API key
- Microphone (for voice input)

## How It Works

1. **Voice Input** → Uses librosa for pitch detection to convert your humming to MIDI notes
2. **Manual Input** → Click on staff or piano keys to place notes
3. **AI Generation** → Sends your melody + prompt to OpenAI/Gemini
4. **Playback** → Web Audio API plays the generated music

## Tips

- Hum clearly and at a steady pace for best voice recognition
- Try different prompts:
  - "Continue with a jazz feel"
  - "Make it more dramatic and build tension"
  - "Add a happy, bouncy ending"
  - "Turn it into a sad ballad"
- Start with 2-4 measures for faster generation

Enjoy creating music with AI! 🎶
