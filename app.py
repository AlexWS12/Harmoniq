"""
Harmoniq - Voice & AI Music Composer
Record your voice/humming or click notes, then AI completes your song!
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import base64
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Store API keys (user will enter via UI)
api_keys = {}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/set_key', methods=['POST'])
def set_api_key():
    """Store user's API key"""
    data = request.json
    provider = data.get('provider', 'openai')
    key = data.get('key', '')
    api_keys[provider] = key
    print(f"✅ API key set for {provider}: {key[:10]}...")
    return jsonify({'success': True})


@app.route('/api/test_ai', methods=['POST'])
def test_ai():
    """Test if AI provider is working"""
    try:
        data = request.json
        provider = data.get('provider', 'openai')
        
        if provider not in api_keys or not api_keys[provider]:
            return jsonify({
                'success': False,
                'error': 'API key not set'
            }), 400
        
        # Simple test
        if provider == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=api_keys[provider])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'test ok'"}],
                max_tokens=10
            )
            return jsonify({'success': True, 'message': 'OpenAI connected successfully'})
        else:
            import google.generativeai as genai
            genai.configure(api_key=api_keys[provider])
            # Try to list models as a test
            models = list(genai.list_models())
            generation_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            print(f"📋 Available Gemini models for generation: {generation_models[:10]}")
            return jsonify({
                'success': True, 
                'message': f'Gemini connected successfully. {len(generation_models)} generation models available',
                'models': generation_models[:5]
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    """Convert recorded audio to musical notes using pitch detection"""
    try:
        audio_data = request.json.get('audio')
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Use librosa or aubio for pitch detection
        import librosa
        
        # Load audio from bytes
        audio_array, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
        
        # Extract pitch using pyin algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_array,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Convert frequencies to MIDI notes
        notes = []
        current_note = None
        note_start = 0
        
        for i, (freq, is_voiced) in enumerate(zip(f0, voiced_flag)):
            if is_voiced and freq > 0:
                midi = librosa.hz_to_midi(freq)
                note_name = librosa.midi_to_note(int(round(midi)))
                
                if current_note != note_name:
                    if current_note:
                        # Save previous note
                        duration = (i - note_start) * 512 / sr  # hop length = 512
                        notes.append({
                            'note': current_note,
                            'midi': int(round(librosa.note_to_midi(current_note))),
                            'start': note_start * 512 / sr,
                            'duration': duration
                        })
                    current_note = note_name
                    note_start = i
        
        # Add last note
        if current_note:
            duration = (len(f0) - note_start) * 512 / sr
            notes.append({
                'note': current_note,
                'midi': int(round(librosa.note_to_midi(current_note))),
                'start': note_start * 512 / sr,
                'duration': duration
            })
        
        return jsonify({
            'success': True,
            'notes': notes
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate_music', methods=['POST'])
def generate_music():
    """Use AI to complete/generate music based on input notes and prompt"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            }), 400
            
        input_notes = data.get('notes', [])
        prompt = data.get('prompt', 'Continue naturally')
        provider = data.get('provider', 'openai')
        num_measures = data.get('measures', 4)
        
        print(f"🎵 Generate request: provider={provider}, notes={len(input_notes)}, prompt={prompt[:50]}")
        
        if provider not in api_keys or not api_keys[provider]:
            return jsonify({
                'success': False,
                'error': 'API key not set. Please enter your API key first.'
            }), 400
        
        # Format input for AI
        melody_description = format_notes_for_ai(input_notes)
        
        # Call AI
        if provider == 'openai':
            generated_notes = generate_with_openai(melody_description, prompt, num_measures, api_keys[provider])
        else:
            generated_notes = generate_with_gemini(melody_description, prompt, num_measures, api_keys[provider])
        
        return jsonify({
            'success': True,
            'notes': generated_notes
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def format_notes_for_ai(notes):
    """Convert notes array to readable format for AI"""
    if not notes:
        return "No input melody - create something original."
    
    melody = []
    for note in notes:
        melody.append(f"{note['note']} ({note['duration']:.2f}s)")
    
    return " -> ".join(melody)


def generate_with_openai(melody, prompt, measures, api_key):
    """Generate music using OpenAI"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        system_prompt = """You are a music composer. Generate melodies in this format:
        
        Return a JSON array of notes:
        [{"note": "C4", "midi": 60, "duration": 0.5, "start": 0.0}, ...]
        
        Where:
        - note: Note name (e.g., "C4", "D#5")
        - midi: MIDI number (60 = C4)
        - duration: Note length in beats (1.0 = quarter, 0.5 = eighth)
        - start: Beat position
        
        Create musically coherent melodies."""
        
        user_prompt = f"""Input melody: {melody}

Task: Generate {measures} measures (16 beats total) that {prompt}

Respond with ONLY the JSON array of notes."""
        
        print(f"🤖 Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        
        # Parse JSON response
        result = response.choices[0].message.content
        print(f"✅ OpenAI response received: {len(result)} chars")
        
        # Extract JSON if wrapped in markdown
        if "```" in result:
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        notes = json.loads(result.strip())
        return notes
    except Exception as e:
        print(f"❌ OpenAI error: {str(e)}")
        raise Exception(f"OpenAI generation failed: {str(e)}")


def generate_with_gemini(melody, prompt, measures, api_key):
    """Generate music using Gemini"""
    try:
        import google.generativeai as genai
        import traceback
        
        print(f"🔑 Configuring Gemini with API key: {api_key[:10]}...{api_key[-4:]}")
        genai.configure(api_key=api_key)
        
        # Dynamically get available models
        print(f"🤖 Fetching available Gemini models...")
        try:
            all_models = list(genai.list_models())
            models_to_try = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
            print(f"   Found {len(models_to_try)} generation models: {models_to_try[:5]}")
            
            if not models_to_try:
                raise ValueError("No generation models available")
        except Exception as e:
            print(f"   ⚠️ Could not fetch models dynamically, using fallback list: {str(e)}")
            # Fallback to common model names
            models_to_try = [
                "gemini-1.5-flash-002",
                "gemini-1.5-pro-002",
                "gemini-2.0-flash-exp"
            ]
        
        last_error = None
        quota_exceeded = False
        for model_name in models_to_try:
            try:
                print(f"  Trying {model_name}...")
                model = genai.GenerativeModel(model_name)
                
                full_prompt = f"""You are a music composer. Generate melodies as JSON arrays.

Input melody: {melody}

Task: Generate {measures} measures (16 beats total) that {prompt}

Format:
[{{"note": "C4", "midi": 60, "duration": 1.0, "start": 0.0}}, ...]

Respond with ONLY the JSON array."""
                
                response = model.generate_content(full_prompt)
                result = response.text
                print(f"✅ Gemini ({model_name}) response: {len(result)} chars")
                print(f"   First 200 chars: {result[:200]}...")
                
                # Parse JSON
                if "```" in result:
                    result = result.split("```")[1]
                    if result.startswith("json"):
                        result = result[4:]
                
                notes = json.loads(result.strip())
                print(f"✅ Successfully parsed {len(notes)} notes from Gemini")
                return notes
                
            except Exception as e:
                last_error = str(e)
                print(f"  ❌ {model_name} failed: {last_error}")
                
                # Check if quota exceeded
                if "429" in last_error or "quota" in last_error.lower():
                    quota_exceeded = True
                    print(f"     ⚠️ Quota limit reached for {model_name}")
                
                print(f"     Traceback: {traceback.format_exc()[:300]}")
                continue
        
        if quota_exceeded:
            raise ValueError(f"Gemini API quota exceeded. Wait a few minutes or check: https://aistudio.google.com/")
        else:
            raise ValueError(f"All Gemini models failed. Last error: {last_error}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Gemini error: {error_msg}")
        if "API_KEY_INVALID" in error_msg or "401" in error_msg or "invalid" in error_msg.lower():
            raise Exception(f"Invalid Gemini API key. Get a free key at: https://aistudio.google.com/app/apikey")
        elif "429" in error_msg or "quota" in error_msg.lower():
            raise Exception(f"Gemini API quota exceeded. The free tier has limits. Wait a few minutes or upgrade at: https://aistudio.google.com/")
        else:
            raise Exception(f"Gemini API error: {error_msg}. Check terminal for details.")


if __name__ == '__main__':
    print("\n🎵 Harmoniq - Voice & AI Music Composer")
    print("=" * 50)
    print("Open your browser to: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
