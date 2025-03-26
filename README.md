本地需下载模型（硬盘中）：放在yj-VTuber路径下

声源检测：pretrained

ASR识别：models

调用api相关端口：
各模型api调用ip端口：
asr：
        url = "http://192.168.1.9:1234/transcribe"）

qwen_vl_llm:
        base_url: 'http://192.168.1.9:11433/v1'
        model: 'Qwen2.5-VL-7B-Instruct'
        temperature: 1.0

gpt_sovits_tts（不可选预训练音色）:
        api_url: 'http://192.168.1.9:2330/tts'

gpt_sovits_tts（可选预训练音色）:
        api_url: 'http://192.168.1.9:3330/tts'

spark_tts:
        api_url: 'http://192.168.1.9:8000'
