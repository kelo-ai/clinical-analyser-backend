"""
Deepgram integration for the clinical transcription module.
Handles both:
  - File-based transcription (REST API) for uploaded audio
  - Live streaming transcription (WebSocket) for real-time recording

Both use diarize=true so segments come back tagged by speaker.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the project root, if present

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError(
        "DEEPGRAM_API_KEY is not set. Create a .env file (see .env.example) "
        "or set it as an environment variable before starting the server."
    )

REST_URL = "https://api.deepgram.com/v1/listen?model=nova-3&language=en&diarize=true&utterances=true&punctuate=true"
WS_URL = "wss://api.deepgram.com/v1/listen?model=nova-3&language=en&diarize=true&utterances=true&punctuate=true&encoding=linear16&sample_rate=16000"


def transcribe_file(audio_bytes: bytes, content_type: str = "audio/*") -> list[dict]:
    """
    Sends a complete audio file to Deepgram's REST API and returns a list
    of diarized segments: [{speaker_label, start_time, end_time, text}, ...]
    """
    print(f"Sending to Deepgram - content_type: {content_type}, size: {len(audio_bytes)} bytes")

    response = requests.post(
        REST_URL,
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": content_type,
        },
        data=audio_bytes,
    )

    print(f"Deepgram response status: {response.status_code}")

    if response.status_code != 200:
        raise RuntimeError(f"Deepgram error {response.status_code}: {response.text}")

    data = response.json()
    return _parse_diarized_response(data)


def _parse_diarized_response(deepgram_response: dict) -> list[dict]:
    """
    Deepgram's diarized response groups speech into 'utterances', each
    tagged with a speaker index. This extracts exactly what we need to
    store as TranscriptSegment rows.
    """
    segments = []
    utterances = deepgram_response.get("results", {}).get("utterances", [])

    for utt in utterances:
        segments.append({
            "speaker_label": f"speaker_{utt['speaker']}",
            "start_time": utt["start"],
            "end_time": utt["end"],
            "text": utt["transcript"],
        })

    return segments