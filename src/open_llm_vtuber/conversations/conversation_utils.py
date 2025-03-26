import asyncio
import datetime
import re
from typing import Optional, Union, Any, List, Dict, Tuple
import numpy as np
import json
from loguru import logger
import os
from ..message_handler import message_handler
from .types import WebSocketSend, BroadcastContext
from .tts_manager import TTSTaskManager
from ..agent.output_types import SentenceOutput, AudioOutput
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..asr.infer_test import VoiceRecognitionManager
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from ..utils.function_tracer import trace_calls
import time

#创建一个 BatchInput 对象，包含用户输入的文本和可选的图像数据。
# Convert class methods to standalone functions
def create_batch_input(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    from_name: str,
) -> BatchInput:
    """Create batch input for agent processing"""
    return BatchInput(
        texts=[
            TextData(source=TextSource.INPUT, content=input_text, from_name=from_name)
        ],
        images=[
            ImageData(
                source=ImageSource(img["source"]),
                data=img["data"],
                mime_type=img["mime_type"],
            )
            for img in (images or [])
        ]
        if images
        else None,
    )

#根据输出类型（文本或音频）调用不同的处理函数
async def process_agent_output(
    output: Union[AudioOutput, SentenceOutput],
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Process agent output with character information and optional translation"""
    output.display_text.name = character_config.character_name
    output.display_text.avatar = character_config.avatar

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            full_response = await handle_sentence_output(
                output,
                live2d_model,
                tts_engine,
                websocket_send,
                tts_manager,
                translate_engine,
            )
        elif isinstance(output, AudioOutput):
            full_response = await handle_audio_output(output, websocket_send)
        else:
            logger.warning(f"Unknown output type: {type(output)}")
    except Exception as e:
        logger.error(f"Error processing agent output: {e}")
        await websocket_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )

    return full_response

#处理文本输出，支持翻译和语音合成。
async def handle_sentence_output(
    output: SentenceOutput,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    async for display_text, tts_text, actions in output:
        logger.debug(f"?? Processing output: '''{tts_text}'''...")

        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)):
                tts_text = translate_engine.translate(tts_text)
            logger.info(f"?? Text after translation: '''{tts_text}'''...")
        else:
            logger.debug("?? No translation engine available. Skipping translation.")

        full_response += display_text.text
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
        )
    return full_response

#处理音频输出，直接发送给客户端。
async def handle_audio_output(
    output: AudioOutput,
    websocket_send: WebSocketSend,
) -> str:
    """Process and send AudioOutput directly to the client"""
    full_response = ""
    async for audio_path, display_text, transcript, actions in output:
        full_response += transcript
        audio_payload = prepare_audio_payload(
            audio_path=audio_path,
            display_text=display_text,
            actions=actions.to_dict() if actions else None,
        )
        await websocket_send(json.dumps(audio_payload))
    return full_response


#发送对话开始
async def send_conversation_start_signals(websocket_send: WebSocketSend) -> None:
    """Send initial conversation signals"""
    await websocket_send(
        json.dumps(
            {
                "type": "control",
                "text": "conversation-chain-start",
            }
        )
    )
    await websocket_send(json.dumps({"type": "full-text", "text": "Thinking..."}))


#处理用户输入，如果是音频，则使用 ASRInterface 的 async_transcribe_np 方法将其转录为文本。
#将转录结果发送给客户端，并返回文本。
# 定义一个函数，将数字转换为类似 Excel 列表的编号格式
def get_user_id(n: int) -> str:
    """Convert a number to an Excel-like column name."""
    user_id = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        user_id = chr(65 + remainder) + user_id
    return user_id

# 定义计数器文件路径
USER_ID_COUNTER_FILE = "user_id_counter.txt"

# 读取计数器文件，获取上次的计数器值
def load_user_id_counter() -> int:
    """Load the user ID counter from a file."""
    if os.path.exists(USER_ID_COUNTER_FILE):
        with open(USER_ID_COUNTER_FILE, "r") as f:
            return int(f.read().strip())
    return 0

# 保存计数器文件，更新计数器值
def save_user_id_counter(counter: int) -> None:
    """Save the user ID counter to a file."""
    with open(USER_ID_COUNTER_FILE, "w") as f:
        f.write(str(counter))

# 获取当前用户计数器值
user_count = load_user_id_counter()

@trace_calls
async def process_audio_parallel(
    audio_data: np.ndarray,
    asr_engine: ASRInterface,
    voice_manager: VoiceRecognitionManager,
    similarity_threshold: float
) -> Tuple[str, str, dict]:
    """并行处理音频的ASR和声源检测"""
    timing_stats = {}
    
    # 创建临时文件
    temp_wav_file = "temp_audio.wav"
    start_time = time.time()
    asr_engine.nparray_to_audio_file(audio_data, sample_rate=asr_engine.SAMPLE_RATE, file_path=temp_wav_file)
    timing_stats['audio_save'] = time.time() - start_time
    
    # 并行执行ASR和声源检测
    start_time = time.time()
    asr_task = asyncio.create_task(asr_engine.async_transcribe_np(audio_data))
    
    # 声源检测在另一个线程中运行，因为它是CPU密集型的
    voice_task = asyncio.get_event_loop().run_in_executor(
        None, 
        voice_manager.process_audio,
        temp_wav_file,
        similarity_threshold
    )
    
    # 等待两个任务完成
    input_text, (score, audio_name) = await asyncio.gather(asr_task, voice_task)
    timing_stats['parallel_processing'] = time.time() - start_time
    
    # 清理临时文件
    start_time = time.time()
    os.remove(temp_wav_file)
    timing_stats['cleanup'] = time.time() - start_time
    
    return input_text, audio_name, timing_stats


# async def process_user_input(
#     user_input: Union[str, np.ndarray],
#     asr_engine: ASRInterface,
#     websocket_send: WebSocketSend,
# ) -> str:
#     """Process user input, converting audio to text if needed"""
#     if isinstance(user_input, np.ndarray):
#         logger.info("Transcribing audio input...")
#         input_text = await asr_engine.async_transcribe_np(user_input)
#         await websocket_send(
#             json.dumps({"type": "user-input-transcription", "text": input_text})
#         )
#         return input_text
#     return user_input

async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
    similarity_threshold: float = 0.55,
    voice_manager: VoiceRecognitionManager = None,
) -> str:
    """处理用户输入，支持并行的ASR和声源检测"""
    if isinstance(user_input, np.ndarray):
        total_start_time = time.time()
        
        # 如果voice_manager未初始化，则初始化它
        if voice_manager is None:
            start_time = time.time()
            comparison_dir = 'C:/Users/20222/3D-Speaker/wav'
            model_id = 'iic/speech_eres2net_sv_zh-cn_16k-common'
            voice_manager = VoiceRecognitionManager(model_id=model_id, comparison_dir=comparison_dir)
            vm_init_time = time.time() - start_time
            logger.info(f"Voice manager initialization time: {vm_init_time:.3f}s")
        
        # 并行处理音频
        input_text, audio_name, timing_stats = await process_audio_parallel(
            user_input, asr_engine, voice_manager, similarity_threshold
        )
        
        
        # 格式化输入文本
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_input = f"{input_text}-----(用户:{audio_name}, 时间：{current_time})------"
        
        # 发送转录结果
        await websocket_send(
            json.dumps({"type": "user-input-transcription", "text": f"用户{audio_name}：{input_text}"})
        )
        
        # 计算总时间并记录所有时间统计
        total_time = time.time() - total_start_time
        timing_stats['total_time'] = total_time
        
        # 记录详细的时间统计
        logger.info("Processing time statistics:")
        logger.info(f"- Audio save time: {timing_stats['audio_save']:.3f}s")
        logger.info(f"- Parallel processing time (ASR + Voice Detection): {timing_stats['parallel_processing']:.3f}s")
        logger.info(f"- Cleanup time: {timing_stats['cleanup']:.3f}s")
        logger.info(f"- Total processing time: {timing_stats['total_time']:.3f}s")
        
        return formatted_input
    return user_input

#完成一次对话轮次，清理资源并发送结束信号。
async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
    broadcast_ctx: Optional[BroadcastContext] = None,
) -> None:
    """Finalize a conversation turn"""
    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        response = await message_handler.wait_for_response(
            client_uid, "frontend-playback-complete"
        )

        if not response:
            logger.warning(f"No playback completion response from {client_uid}")
            return

    await websocket_send(json.dumps({"type": "force-new-message"}))

    if broadcast_ctx and broadcast_ctx.broadcast_func:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            {"type": "force-new-message"},
            broadcast_ctx.current_client_uid,
        )

    await send_conversation_end_signal(websocket_send, broadcast_ctx)


#结束的信号
async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    broadcast_ctx: Optional[BroadcastContext],
    session_emoji: str = "??",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }

    await websocket_send(json.dumps(chain_end_msg))

    if broadcast_ctx and broadcast_ctx.broadcast_func and broadcast_ctx.group_members:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            chain_end_msg,
        )

    logger.info(f"????? Conversation Chain {session_emoji} completed!")


#清理对话资源，例如清除 TTS 任务列表。
def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    tts_manager.clear()
    logger.debug(f"?? Clearing up conversation {session_emoji}.")

EMOJI_LIST = [
    "🐶",
    "🐱",
    "🐭",
    "🐹",
    "🐰",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐮",
    "🐷",
    "🐸",
    "🐵",
    "🐔",
    "🐧",
    "🐦",
    "🐤",
    "🐣",
    "🐥",
    "🦆",
    "🦅",
    "🦉",
    "🦇",
    "🐺",
    "🐗",
    "🐴",
    "🦄",
    "🐝",
    "🌵",
    "🎄",
    "🌲",
    "🌳",
    "🌴",
    "🌱",
    "🌿",
    "☘️",
    "🍀",
    "🍂",
    "🍁",
    "🍄",
    "🌾",
    "💐",
    "🌹",
    "🌸",
    "🌛",
    "🌍",
    "⭐️",
    "🔥",
    "🌈",
    "🌩",
    "⛄️",
    "🎃",
    "🎄",
    "🎉",
    "🎏",
    "��",
    "🀄️",
    "🎭",
    "🎨",
    "🧵",
    "🪡",
    "🧶",
    "🥽",
    "🥼",
    "🦺",
    "👔",
    "👕",
    "👜",
    "👑",
]
