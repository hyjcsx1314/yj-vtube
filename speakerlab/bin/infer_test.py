import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio
import logging
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import random

logging.basicConfig(level=logging.ERROR)  # 只输出错误信息
try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

# from modelscope.hub.snapshot_download import snapshot_download
# from modelscope.pipelines.util import is_official_hub_path

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2NetV2_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 26,
        'scale': 2,
        'expansion': 2,
    },
}

ERes2NetV2_w24s4ep4_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 24,
        'scale': 4,
        'expansion': 4,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

ECAPA_CNCeleb = {
    'obj': 'speakerlab.models.ecapa_tdnn.ECAPA_TDNN.ECAPA_TDNN',
    'args': {
        'input_size': 80,
        'lin_neurons': 192,
        'channels': [1024, 1024, 1024, 1024, 3072],
    },
}

supports = {
    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    # ERes2Net trained on 200k labeled speakers
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    # ERes2NetV2 trained on 200k labeled speakers
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_COMMON,
        'model_pt': 'pretrained_eres2netv2.ckpt',
    },
    # ERes2NetV2_w24s4ep4 trained on 200k labeled speakers
    'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_w24s4ep4_COMMON,
        'model_pt': 'pretrained_eres2netv2w24s4ep4.ckpt',
    },
    # ERes2Net_Base trained on 200k labeled speakers
    'iic/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
    },
    # CAM++ trained on a large-scale Chinese-English corpus
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
    # CAM++ trained on VoxCeleb
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    # ERes2Net trained on VoxCeleb
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    # ERes2Net_Base trained on 3dspeaker
    'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1', 
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    # ERes2Net_large trained on 3dspeaker
    'iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
    # ECAPA-TDNN trained on CNCeleb
    'iic/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on 3dspeaker
    'iic/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa-tdnn.ckpt',
    },
    # ECAPA-TDNN trained on VoxCeleb
    'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k': {
        'revision': 'v1.0.1', 
        'model': ECAPA_CNCeleb,
        'model_pt': 'ecapa_tdnn.bin',
    },
}

# 生成规律的命名
def generate_name(counter):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    name = ""
    while counter > 0:
        counter -= 1
        name = alphabet[counter % 26] + name
        counter //= 26
    return name


# 获取当前的命名状态
def get_current_counter(counter_file):
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            return int(f.read().strip())
    return 2


# 更新命名状态
def update_counter(counter_file, counter):
    with open(counter_file, "w") as f:
        f.write(str(counter))

def load_wav(wav_file, obj_fs=16000):
    wav, fs = torchaudio.load(wav_file)
    if fs != obj_fs:
        wav = torchaudio.transforms.Resample(fs, obj_fs)(wav)
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    return wav

def compute_embedding(wav_file, model, feature_extractor, device):
    wav = load_wav(wav_file)
    feat = feature_extractor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(feat).detach().squeeze(0).cpu().numpy()
    return embedding

def record_audio(duration=5, fs=16000):
    print("Recording audio for 5 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    return recording

def save_audio(recording, fs=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav.write(tmpfile.name, fs, recording)
        return tmpfile.name

def main():
    parser = argparse.ArgumentParser(description='Extract speaker embeddings and compute similarity scores.')
    parser.add_argument('--model_id', default='iic/speech_eres2net_sv_zh-cn_16k-common', type=str, help='Model id')
    parser.add_argument('--comparison_dir', default='C:/Users/20222/3D-Speaker/wav', type=str, help='Directory containing comparison audio files')
    parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
    parser.add_argument('--model_path',default='C:/Users/20222/.cache/modelscope/hub/models/iic/speech_eres2netv2_sv_zh-cn_16k-common', type=str, help='Path to the local model file')
    args = parser.parse_args()

    # 加载模型
    conf = supports[args.model_id]
    model_path = args.model_path or os.path.join(args.local_model_dir, conf['model_pt'])
    model = dynamic_import(conf['model']['obj'])(**conf['model']['args'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    # 录制音频
    recording = record_audio(duration=5, fs=16000)
    target_wav = save_audio(recording, fs=16000)

    # 计算目标音频的嵌入
    target_embedding = compute_embedding(target_wav, model, feature_extractor, device)

    # 获取所有对比音频文件
    comparison_wavs = [os.path.join(args.comparison_dir, f) for f in os.listdir(args.comparison_dir) if f.endswith('.wav')]
    if not comparison_wavs:
        raise Exception(f'[ERROR]: No .wav files found in {args.comparison_dir}')

    # 计算相似度
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    max_score = -1
    best_match = None

    for wav in comparison_wavs:
        embedding = compute_embedding(wav, model, feature_extractor, device)
        score = similarity(torch.from_numpy(target_embedding).unsqueeze(0), torch.from_numpy(embedding).unsqueeze(0)).item()
        if score > max_score:
            max_score = score
            best_match = wav

    print(f'Highest similarity score: {max_score:.4f} with {best_match} (Target: {target_wav})')

    # 设置相似度得分阈值
    threshold = 0.60

    # 获取当前的命名状态
    counter_file = os.path.join(args.comparison_dir, "counter.txt")
    counter = get_current_counter(counter_file)

    # 如果最高相似度得分低于阈值，则将录制的音频保存到音频库中
    if max_score < threshold:
        user_name = generate_name(counter)
        new_audio_path = os.path.join(args.comparison_dir, f"user_{user_name}.wav")
        os.rename(target_wav, new_audio_path)
        print(f"New audio saved to {new_audio_path}")
        # 更新命名状态
        update_counter(counter_file, counter + 1)
    else:
        os.remove(target_wav)


if __name__ == '__main__':
    main()