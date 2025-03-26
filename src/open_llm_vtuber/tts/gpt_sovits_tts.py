####
# change from xTTS.py
####

import re
import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://192.168.1.9:2330/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut5",
        batch_size: str = "3",
        media_type: str = "wav",
        streaming_mode: str = "false",
        voice_preset: str = "libing",  # 预设音色
        top_k: int = 3,
        top_p: float = 0.8,
        temperature: float = 0.7,
        repetition_penalty: float = 5.0,
        speed_factor: float = 1.0,
        batch_threshold: float = 0.9,
        split_bucket: bool = True,
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type
        self.streaming_mode = streaming_mode
        self.voice_preset = voice_preset
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.speed_factor = speed_factor
        self.batch_threshold = batch_threshold
        self.split_bucket = split_bucket

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Generate audio from text using GPT-SoVITS API.
        
        Args:
            text: Text to convert to speech
            file_name_no_ext: Name of the file without extension
            
        Returns:
            str: Path to the generated audio file
        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        cleaned_text = re.sub(r"\[.*?\]", "", text)
        
        # 准备请求数据
        request_data = {
            "text": cleaned_text,
            "text_lang": self.text_lang,
            "voice_preset": self.voice_preset,  # 使用预设音色
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "speed_factor": self.speed_factor,
            "text_split_method": self.text_split_method,
            "batch_size": int(self.batch_size),
            "batch_threshold": self.batch_threshold,
            "split_bucket": self.split_bucket
        }

        try:
            # 发送 POST 请求到 TTS API
            response = requests.post(self.api_url, json=request_data, timeout=120)
            
            # 检查请求是否成功
            response.raise_for_status()
            
            # 获取音频统计信息
            audio_stats = response.headers.get('X-Audio-Stats', '{}')
            logger.debug(f"Audio statistics: {audio_stats}")
            
            # 保存音频内容到文件
            with open(file_name, "wb") as audio_file:
                audio_file.write(response.content)
                
            return file_name
            
        except requests.exceptions.RequestException as e:
            logger.critical(f"Error: Failed to generate audio. Error: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.critical(f"Server response: {e.response.text}")
            raise
