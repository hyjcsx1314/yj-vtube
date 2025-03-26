from typing import Type
from .tts_interface import TTSInterface
from loguru import logger


class TTSFactory:
    @staticmethod
    def get_tts_engine(engine_type, **kwargs) -> Type[TTSInterface]:
        if engine_type == "azure_tts":
            from .azure_tts import TTSEngine as AzureTTSEngine

            return AzureTTSEngine(
                kwargs.get("api_key"),
                kwargs.get("region"),
                kwargs.get("voice"),
                kwargs.get("pitch"),
                kwargs.get("rate"),
            )
        elif engine_type == "bark_tts":
            from .bark_tts import TTSEngine as BarkTTSEngine

            return BarkTTSEngine(kwargs.get("voice"))
        elif engine_type == "edge_tts":
            from .edge_tts import TTSEngine as EdgeTTSEngine

            return EdgeTTSEngine(kwargs.get("voice"))
        elif engine_type == "pyttsx3_tts":
            from .pyttsx3_tts import TTSEngine as Pyttsx3TTSEngine

            return Pyttsx3TTSEngine()
        elif engine_type == "cosyvoice_tts":
            from .cosyvoice_tts import TTSEngine as CosyvoiceTTSEngine

            return CosyvoiceTTSEngine(
                client_url=kwargs.get("client_url"),
                mode_checkbox_group=kwargs.get("mode_checkbox_group"),
                sft_dropdown=kwargs.get("sft_dropdown"),
                prompt_text=kwargs.get("prompt_text"),
                prompt_wav_upload_url=kwargs.get("prompt_wav_upload_url"),
                prompt_wav_record_url=kwargs.get("prompt_wav_record_url"),
                instruct_text=kwargs.get("instruct_text"),
                seed=kwargs.get("seed"),
                api_name=kwargs.get("api_name"),
            )
        elif engine_type == "cosyvoice2_tts":
            from .cosyvoice2_tts import TTSEngine as Cosyvoice2TTSEngine

            return Cosyvoice2TTSEngine(
                client_url=kwargs.get("client_url"),
                mode_checkbox_group=kwargs.get("mode_checkbox_group"),
                sft_dropdown=kwargs.get("sft_dropdown"),
                prompt_text=kwargs.get("prompt_text"),
                prompt_wav_upload_url=kwargs.get("prompt_wav_upload_url"),
                prompt_wav_record_url=kwargs.get("prompt_wav_record_url"),
                instruct_text=kwargs.get("instruct_text"),
                stream=kwargs.get("stream"),
                seed=kwargs.get("seed"),
                speed=kwargs.get("speed"),
                api_name=kwargs.get("api_name"),
            )
        elif engine_type == "melo_tts":
            from .melo_tts import TTSEngine as MeloTTSEngine

            return MeloTTSEngine(
                speaker=kwargs.get("speaker"),
                language=kwargs.get("language"),
                device=kwargs.get("device"),
                speed=kwargs.get("speed"),
            )
        elif engine_type == "x_tts":
            from .x_tts import TTSEngine as XTTSEngine

            return XTTSEngine(
                api_url=kwargs.get("api_url"),
                speaker_wav=kwargs.get("speaker_wav"),
                language=kwargs.get("language"),
            )
        elif engine_type == "gpt_sovits_tts":
            from .gpt_sovits_tts import TTSEngine as GSVEngine

            # 记录所有传入的参数
            logger.debug(f"Received kwargs for GPT-SoVITS: {kwargs}")
            
            # 获取配置参数
            config = kwargs.get("gpt_sovits_tts", {})
            if not config:
                logger.warning("No gpt_sovits_tts config found in kwargs")
                config = kwargs  # 如果没有找到配置，使用直接传入的参数
            
            # 获取并验证 voice_preset
            voice_preset = config.get("voice_preset")
            logger.debug(f"voice_preset value: {voice_preset}, type: {type(voice_preset)}")
            
            if not voice_preset:
                logger.error("voice_preset is empty or None")
                raise ValueError("voice_preset must be a valid string")
            
            if not isinstance(voice_preset, str):
                logger.error(f"voice_preset is not a string: {type(voice_preset)}")
                raise ValueError("voice_preset must be a valid string")

            try:
                # 确保 batch_size 是字符串类型
                batch_size = str(config.get("batch_size", "3"))
                
                return GSVEngine(
                    api_url=config.get("api_url", "http://192.168.1.9:2330/tts"),
                    text_lang=config.get("text_lang", "zh"),
                    voice_preset=voice_preset,
                    top_k=config.get("top_k", 3),
                    top_p=config.get("top_p", 0.8),
                    temperature=config.get("temperature", 0.7),
                    repetition_penalty=config.get("repetition_penalty", 5.0),
                    speed_factor=config.get("speed_factor", 1.0),
                    text_split_method=config.get("text_split_method", "cut5"),
                    batch_size=batch_size,
                    batch_threshold=config.get("batch_threshold", 0.9),
                    split_bucket=config.get("split_bucket", True),
                )
            except Exception as e:
                logger.error(f"Failed to create GPT-SoVITS engine: {str(e)}")
                logger.error(f"Config used: {config}")
                raise
        elif engine_type == "spark_tts":
            from .spark_tts import SparkTTS

            return SparkTTS(
                api_url=kwargs.get("api_url"),
                ref_audio_path=kwargs.get("ref_audio_path"),
                prompt_text=kwargs.get("prompt_text"),
            )
        elif engine_type == "coqui_tts":
            from .coqui_tts import TTSEngine as CoquiTTSEngine

            return CoquiTTSEngine(
                model_name=kwargs.get("model_name"),
                speaker_wav=kwargs.get("speaker_wav"),
                language=kwargs.get("language"),
                device=kwargs.get("device"),
            )

        elif engine_type == "fish_api_tts":
            from .fish_api_tts import TTSEngine as FishAPITTSEngine

            return FishAPITTSEngine(
                api_key=kwargs.get("api_key"),
                reference_id=kwargs.get("reference_id"),
                latency=kwargs.get("latency"),
                base_url=kwargs.get("base_url"),
            )
        elif engine_type == "sherpa_onnx_tts":
            from .sherpa_onnx_tts import TTSEngine as SherpaOnnxTTSEngine

            return SherpaOnnxTTSEngine(**kwargs)

        else:
            raise ValueError(f"Unknown TTS engine type: {engine_type}")


# Example usage:
# tts_engine = TTSFactory.get_tts_engine("azure", api_key="your_api_key", region="your_region", voice="your_voice")
# tts_engine.speak("Hello world")
