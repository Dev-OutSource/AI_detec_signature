import json
import os
import requests
from typing import Optional
from pydantic import BaseModel

class TTSResponse(BaseModel):
    audio_file: Optional[bytes] = None
    status: int

    class Config:
        arbitrary_types_allowed = True  # Allow Response type

class TTSViettelAI:
    def __init__(self):
        self.configs = {
            'api_url': "https://viettelai.vn/tts/speech_synthesis",
            'voice': "hn-thanhtung",
            'speed': 1.0,
            'without_filter': True
        }
        self.url = self.configs["api_url"]

    def _create_payload(self, text: str) -> str:
        """Create the payload for the TTS API request"""
        return json.dumps(
            {
                "text": text,
                "voice": self.configs["voice"],
                "speed": self.configs["speed"],
                "tts_return_option": 2,
                "token": os.getenv("VIETTEL_AI_KEY"),
                "without_filter": self.configs["without_filter"],
            }
        )

    def _make_api_request(self, payload: str) -> requests.Response:
        """Make the API request to Viettel TTS service"""
        headers = {"accept": "*/*", "Content-Type": "application/json"}
        return requests.post(self.url, headers=headers, data=payload)

    def handle(self, text: str) -> TTSResponse:
        """
        Handle TTS request and return audio response
        """
        try:
            # Get API response
            payload = self._create_payload(text)
            response = self._make_api_request(payload)
            print("---------", response.status_code)
            if response.status_code != 200:
                return TTSResponse(audio_file=None, status=response.status_code)

            return TTSResponse(
                audio_file=response.content,  # .decode('ISO-8859-1'),
                status=200,
            )

        except Exception as error:
            print(f"TTS processing failed: {str(error)}")
            return TTSResponse(audio_file=None, status=500)