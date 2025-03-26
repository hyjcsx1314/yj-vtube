
from typing import Type
from .asr_interface import ASRInterface


class ASRFactory:
    @staticmethod
    def get_asr_system(system_name: str, **kwargs) -> Type[ASRInterface]:
        if system_name == "faster_whisper":
            from .faster_whisper_asr import VoiceRecognition as FasterWhisperASR

            return FasterWhisperASR(
                model_path=kwargs.get("model_path"),
                download_root=kwargs.get("download_root"),
                language=kwargs.get("language"),
                device=kwargs.get("device"),
            )
        elif system_name == "whisper_cpp":
            from .whisper_cpp_asr import VoiceRecognition as WhisperCPPASR

            return WhisperCPPASR(**kwargs)
        elif system_name == "whisper":
            from .openai_whisper_asr import VoiceRecognition as WhisperASR

            return WhisperASR(**kwargs)
        elif system_name == "fun_asr":
            from .fun_asr import VoiceRecognition as FunASR

            return FunASR(
                model_name=kwargs.get("model_name"),
                vad_model=kwargs.get("vad_model"),
                punc_model=kwargs.get("punc_model"),
                ncpu=kwargs.get("ncpu"),
                hub=kwargs.get("hub"),
                device=kwargs.get("device"),
                language=kwargs.get("language"),
                use_itn=kwargs.get("use_itn"),
                # sample_rate=kwargs.get("sample_rate"),
            )
        elif system_name == "azure_asr":
            from .azure_asr import VoiceRecognition as AzureASR

            return AzureASR(
                subscription_key=kwargs.get("api_key"),
                region=kwargs.get("region"),
                languages=kwargs.get("languages", ["en-US", "zh-CN"]),
            )
        elif system_name == "groq_whisper_asr":
            from .groq_whisper_asr import VoiceRecognition as GroqWhisperASR

            return GroqWhisperASR(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model"),
                lang=kwargs.get("lang"),
            )
        elif system_name == "sherpa_onnx_asr":
            from .sherpa_onnx_asr import VoiceRecognition as SherpaOnnxASR

            return SherpaOnnxASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR system: {system_name}")


# from typing import Type
# from .asr_interface import ASRInterface
# from .sherpa_onnx_asr import VoiceRecognition
# from .sherpa_http_asr import SherpaHttpASR

# class ASRFactory:
#     @staticmethod
#     def get_asr_system(system_name: str, **kwargs) -> ASRInterface:
#         """
#         根据系统名称返回相应的ASR系统实例
#         """
#         if system_name == "faster_whisper":
#             from .faster_whisper_asr import FasterWhisperASR
#             return FasterWhisperASR(**kwargs)
#         elif system_name == "whisper_cpp":
#             from .whisper_cpp_asr import WhisperCppASR
#             return WhisperCppASR(**kwargs)
#         elif system_name == "whisper":
#             from .whisper_asr import WhisperASR
#             return WhisperASR(**kwargs)
#         elif system_name == "fun_asr":
#             from .fun_asr import FunASR
#             return FunASR(**kwargs)
#         elif system_name == "azure_asr":
#             from .azure_asr import AzureASR
#             return AzureASR(**kwargs)
#         elif system_name == "groq_whisper_asr":
#             from .groq_whisper_asr import GroqWhisperASR
#             return GroqWhisperASR(**kwargs)
#         elif system_name == "sherpa_onnx_asr":
#             # 检查是否使用HTTP API
#             if "api_url" in kwargs:
#                 # 移除不需要的参数
#                 http_kwargs = {k: v for k, v in kwargs.items() if k in ["api_url", "num_threads", "use_itn", "provider"]}
#                 return SherpaHttpASR(**http_kwargs)
#             else:
#                 # 移除HTTP API相关的参数
#                 local_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_url"]}
#                 return VoiceRecognition(**local_kwargs)
#         else:
#             raise ValueError(f"Unknown ASR system: {system_name}")
