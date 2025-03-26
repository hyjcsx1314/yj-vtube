import requests
import json
import time
import wave
import numpy as np
from io import BytesIO
import os
import sounddevice as sd

# API基础URL
BASE_URL = "http://192.168.1.9:2330"

def wait_for_model_ready(max_retries=30, retry_interval=2):
    """等待模型加载完成"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/status")
            if response.status_code == 200:
                print("Model is ready")
                return True
        except:
            pass
        print(f"Waiting for model to be ready... ({i+1}/{max_retries})")
        time.sleep(retry_interval)
    return False

def play_audio(audio_array, sample_rate):
    """播放音频数据"""
    # 将音频数据转换为float32格式，并归一化到[-1, 1]范围
    audio_float = audio_array.astype(np.float32) / np.iinfo(np.int16).max
    # 播放音频
    sd.play(audio_float, sample_rate)
    # 等待音频播放完成
    sd.wait()

def test_tts_with_preset(text, voice_preset):
    """测试指定文本和音色的TTS生成"""
    print(f"\nTesting TTS with preset '{voice_preset}' and text: {text}")
    
    # 准备请求数据
    request_data = {
        "text": text,
        "text_lang": "zh",
        "voice_preset": voice_preset,  # 直接使用预设音色
        "top_k": 3,
        "top_p": 0.8,
        "temperature": 0.7,
        "repetition_penalty": 5.0,
        "speed_factor": 1.0,
        "text_split_method": "cut5",
        "batch_size": 3,
        "batch_threshold": 0.9,
        "split_bucket": True
    }
    
    try:
        # 发送请求并记录时间
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/tts", json=request_data)
        end_time = time.time()
        
        if response.status_code == 200:
            # 获取音频统计信息
            audio_stats = json.loads(response.headers.get('X-Audio-Stats', '{}'))
            print(f"Request completed in {end_time - start_time:.2f} seconds")
            print(f"Audio statistics: {json.dumps(audio_stats, indent=2)}")
            
            # 读取WAV文件数据
            wav_data = BytesIO(response.content)
            with wave.open(wav_data, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / frame_rate
                
                print(f"\nWAV file details:")
                print(f"Channels: {n_channels}")
                print(f"Sample width: {sample_width} bytes")
                print(f"Frame rate: {frame_rate} Hz")
                print(f"Number of frames: {n_frames}")
                print(f"Duration: {duration:.2f} seconds")
                
                # 读取音频数据进行分析
                raw_data = wav_file.readframes(n_frames)
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
                
                # 计算信噪比
                signal_power = np.mean(audio_array[audio_array != 0] ** 2)
                noise_power = np.mean(audio_array[audio_array == 0] ** 2)
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    print(f"Signal-to-Noise Ratio: {snr:.2f} dB")
                
                # 播放音频
                print("\nPlaying audio...")
                play_audio(audio_array, frame_rate)
                print("Audio playback completed")
            
            return True
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # 等待模型加载
    if not wait_for_model_ready():
        print("Model failed to load")
        exit(1)
    
    # 测试长文本
    test_text = """
    春天来了，樱花开满了整个公园。微风轻拂，粉色的花瓣随风飘舞，像是一场浪漫的舞会。
    小朋友们在草地上奔跑嬉戏，欢声笑语回荡在空中。远处的湖面波光粼粼，倒映着蓝天白云。
    这是一个完美的春日午后，让人感到无比的愉悦和放松。让我们一起享受这美好的时光吧！
    """
    
    voice_preset = "libing"  # 使用预设音色
    test_tts_with_preset(test_text, voice_preset)