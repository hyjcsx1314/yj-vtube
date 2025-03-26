import abc
import numpy as np
import asyncio
from pydub import AudioSegment
import os
import uuid
import base64
import hashlib
import json
import requests
import time
import datetime
from ..utils.function_tracer import trace_calls

class ASRInterface(metaclass=abc.ABCMeta):
    SAMPLE_RATE = 16000
    NUM_CHANNELS = 1
    SAMPLE_WIDTH = 2

    async def async_transcribe_np(self, audio: np.ndarray) -> str:
        """Asynchronously transcribe speech audio in numpy array format.

        By default, this runs the synchronous transcribe_np in a coroutine.
        Subclasses can override this method to provide true async implementation.

        Args:
            audio: The numpy array of the audio data to transcribe.

        Returns:
            str: The transcription result.
        """
        return await asyncio.to_thread(self.transcribe_np, audio)

    @abc.abstractmethod
    def transcribe_np(self, audio: np.ndarray) -> str:
        """Transcribe speech audio in numpy array format and return the transcription.

        Args:
            audio: The numpy array of the audio data to transcribe.
        """
        raise NotImplementedError
    @trace_calls
    def nparray_to_audio_file(
        self, audio: np.ndarray, sample_rate: int, file_path: str
    ) -> None:
        """Convert a numpy array of audio data to a .wav file.

        Args:
            audio: The numpy array of audio data.
            sample_rate: The sample rate of the audio data.
            file_path: The path to save the .wav file.
        """
        import wave

        # Make sure the audio is in the range [-1, 1]
        audio = np.clip(audio, -1, 1)
        # Convert the audio to 16-bit PCM
        audio_integer = (audio * 32767).astype(np.int16)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_integer.tobytes())


    def process_wav_to_mp3(self, input_file: str, output_file: str, max_duration: float = 4.0) -> None:
        """Process a WAV audio file and convert it to MP3 format with a maximum duration.

        Args:
            input_file: The path to the input WAV file.
            output_file: The path to save the processed MP3 file.
            max_duration: The maximum duration of the output audio in seconds (default: 4.0).
        """
        # Load the WAV file
        audio = AudioSegment.from_wav(input_file)
        
        # Set the desired audio properties
        audio = audio.set_frame_rate(self.SAMPLE_RATE)
        audio = audio.set_channels(self.NUM_CHANNELS)
        audio = audio.set_sample_width(self.SAMPLE_WIDTH)
        
        # Check if the audio duration exceeds the maximum allowed duration
        if len(audio) > max_duration * 1000:  # Duration in milliseconds
            # logger.warning(f"Audio duration exceeds {max_duration} seconds. Trimming audio...")
            audio = audio[:int(max_duration * 1000)]  # Trim to max_duration
        
        # Export the processed audio to MP3
        audio.export(output_file, format="mp3")

    def add_audio_to_library(self, audio_file: str, group_id: str, feature_id: str, feature_info: str) -> dict:
        """Add processed audio to the voiceprint library.

        Args:
            audio_file: The path to the processed audio file.
            group_id: The group ID for the voiceprint library.
            feature_id: The feature ID for the audio.
            feature_info: Additional information about the audio feature.

        Returns:
            dict: The response from the voiceprint library API.
        """
        with open(audio_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        timestamp = str(int(time.time() * 1000))
        nonce = str(uuid.uuid1()).replace('-', '')
        sign = self.get_sign(timestamp, nonce)

        headers = {"Content-Type": "application/json"}
        create_feature_param = {
            "appkey": appkey,
            "timestamp": timestamp,
            "nonce": nonce,
            "sign": sign,
            "groupId": group_id,
            "featureId": feature_id,
            "featureInfo": feature_info,
            "audioData": audio_data,
            "audioSampleRate": self.SAMPLE_RATE,
            "audioFormat": "mp3"  # 修改为 MP3 格式
        }
        #print('create_feature_param', create_feature_param)
        create_feature_resp = requests.post(url=host + createFeatureEndPoint, data=json.dumps(create_feature_param),
                                            headers=headers)
        create_feature_result = json.loads(create_feature_resp.content)
        # print('create_feature_result', create_feature_result)
        # print('------------------------------------------------------------')

        return create_feature_result

    def identify_feature_by_group_id(self, audio_file: str, group_id: str) -> dict:
        """Identify the audio feature by group ID."""
        with open(audio_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        timestamp = str(int(time.time() * 1000))
        nonce = str(uuid.uuid1()).replace('-', '')
        sign = self.get_sign(timestamp, nonce)
        headers = {"Content-Type": "application/json"}
        identify_feature_param = {
            "appkey": appkey,
            "timestamp": timestamp,
            "nonce": nonce,
            "sign": sign,
            "groupId": group_id,
            "topN": 1,
            "audioData": audio_data,
            "audioSampleRate": self.SAMPLE_RATE,
            "audioFormat": "mp3"
        }
        identify_feature_resp = requests.post(url=host + identifyFeatureByGroupIdEndPoint,
                                              data=json.dumps(identify_feature_param),
                                              headers=headers)
        identify_feature_result = json.loads(identify_feature_resp.content)
        # print('identify_feature_result', identify_feature_result)
        # print('------------------------------------------------------------')
        
        # 确保返回结果
        return identify_feature_result

    @staticmethod
    def get_sign(timestamp: str, nonce: str) -> str:
        hs = hashlib.sha256()
        hs.update((appkey + timestamp + secret + nonce).encode('utf-8'))
        signature = hs.hexdigest().upper()
        return signature
    
host = 'https://ai-vpr.hivoice.cn'
appkey = "qglq5vun6w4sqmsut27an4pnp3zamgazz4ciahqb"
secret = '1f3dda416e4fec67aa41e88c9f31b626'
createGroupEndPoint = '/vpr/v1/createGroup'
createFeatureEndPoint = '/vpr/v1/createFeature'
confirmFeatureEndPoint = '/vpr/v1/confirmFeature'
identifyFeatureByGroupIdEndPoint = '/vpr/v1/identifyFeatureByGroupId'
identifyFeatureByIdsEndPoint = '/vpr/v1/identifyFeatureByIds'
updateFeatureByIdEndPoint = '/vpr/v1/updateFeatureById'
delFeatureByIdEndPoint = '/vpr/v1/delFeatureById'
delGroupByIdEndPoint = '/vpr/v1/delGroupById'
findFeatureListByGroupIdEndPoint = '/vpr/v1/findFeatureListByGroupId'