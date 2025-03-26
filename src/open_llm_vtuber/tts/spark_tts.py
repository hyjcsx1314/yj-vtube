import os
import requests
from typing import Optional
from .tts_interface import TTSInterface

class SparkTTS(TTSInterface):
    def __init__(self, api_url: str = 'http://localhost:8000', ref_audio_path: str = None, prompt_text: str = None):
        """
        Initialize SparkTTS with configuration.
        
        Args:
            api_url: The base URL of the SparkTTS API
            ref_audio_path: Path to the reference audio file
            prompt_text: Text corresponding to the reference audio (optional)
        """
        self.api_url = api_url
        self.ref_audio_path = ref_audio_path
        self.prompt_text = prompt_text
        
        if not self.ref_audio_path:
            raise ValueError("ref_audio_path is required for SparkTTS")
            
        if not os.path.exists(self.ref_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {self.ref_audio_path}")

    def generate_audio(self, text: str, output_path: str) -> str:
        """
        Generate audio from text using SparkTTS.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the output audio file
            
        Returns:
            str: Path to the generated audio file
        """
        return self.text_to_speech(text, output_path)

    def text_to_speech(self, text: str, output_path: str) -> str:
        """
        Convert text to speech using SparkTTS voice cloning.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the output audio file
            
        Returns:
            str: Path to the generated audio file
        """
        # Prepare the API endpoint URL
        url = f"{self.api_url.rstrip('/')}/clone_voice"
        
        # Prepare the files and data for the request
        files = {
            'prompt_audio': ('reference.wav', open(self.ref_audio_path, 'rb'), 'audio/wav'),
        }
        
        data = {
            'text': text,
        }
        
        if self.prompt_text:
            data['prompt_text'] = self.prompt_text
        
        try:
            # Send POST request to the API
            response = requests.post(url, files=files, data=data)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Save the response content as an audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while making the request: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Server response: {e.response.text}")
            raise
            
        finally:
            # Close the file
            files['prompt_audio'][1].close() 