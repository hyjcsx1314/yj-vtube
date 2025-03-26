import os
import tempfile
import requests
import numpy as np
import soundfile as sf
from loguru import logger
from .sherpa_onnx_asr import VoiceRecognition
from .asr_interface import ASRInterface

class ASRFactory:
    @staticmethod
    def get_asr_system(system_name: str, **kwargs) -> ASRInterface:
        """
        根据系统名称返回相应的ASR系统实例
        """
        # if system_name == "faster_whisper":
        #     from .faster_whisper_asr import FasterWhisperASR
        #     return FasterWhisperASR(**kwargs)
        # elif system_name == "whisper_cpp":
        #     from .whisper_cpp_asr import WhisperCppASR
        #     return WhisperCppASR(**kwargs)
        # elif system_name == "whisper":
        #     from .whisper_asr import WhisperASR
        #     return WhisperASR(**kwargs)
        # elif system_name == "fun_asr":
        #     from .fun_asr import FunASR
        #     return FunASR(**kwargs)
        # elif system_name == "azure_asr":
        #     from .azure_asr import AzureASR
        #     return AzureASR(**kwargs)
        # elif system_name == "groq_whisper_asr":
        #     from .groq_whisper_asr import GroqWhisperASR
        #     return GroqWhisperASR(**kwargs)
        if system_name == "sherpa_onnx_asr":
            # 检查是否使用HTTP API
            if "api_url" in kwargs:
                return SherpaHttpASR(**kwargs)
            else:
                return VoiceRecognition(**kwargs)
        else:
            raise ValueError(f"Unknown ASR system: {system_name}")




class SherpaHttpASR(ASRInterface):
    def __init__(self, api_url: str = "http://localhost:1234/transcribe", 
                 num_threads: int = 4, 
                 use_itn: bool = True, 
                 provider: str = "cpu") -> None:
        self.api_url = api_url
        self.num_threads = num_threads
        self.use_itn = use_itn
        self.provider = provider
        self.SAMPLE_RATE = 16000  # 固定采样率
        logger.info(f"初始化Sherpa-HTTP-ASR: API地址={self.api_url}, 线程数={self.num_threads}, ITN={self.use_itn}, 推理设备={self.provider}")

    def transcribe_np(self, audio_data: np.ndarray) -> str:
        """转写numpy格式的音频数据"""
        logger.info("开始转写...")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            logger.debug(f"创建临时音频文件: {temp_path}")
            sf.write(temp_path, audio_data, self.SAMPLE_RATE)

        try:
            # 发送HTTP请求
            logger.info(f"发送HTTP请求到 {self.api_url}")
            files = {"file": open(temp_path, "rb")}
            response = requests.post(self.api_url, files=files)
            files["file"].close()

            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "")
                logger.info(f"转写成功: {text}")
                return text
            else:
                logger.error(f"API请求失败: 状态码={response.status_code}, 响应={response.text}")
                return ""

        except Exception as e:
            logger.error(f"转写过程发生错误: {str(e)}")
            return ""
        
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"删除临时音频文件: {temp_path}") 