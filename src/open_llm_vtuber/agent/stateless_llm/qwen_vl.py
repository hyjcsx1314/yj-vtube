from typing import AsyncIterator, List, Dict, Any
import aiohttp
import json
from loguru import logger
from PIL import Image
import base64
import io
import re

from .stateless_llm_interface import StatelessLLMInterface

def decode_base64_image(base64_string: str) -> Image.Image:
    """将base64字符串解码为PIL Image对象"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

class QwenVLLLM(StatelessLLMInterface):
    def __init__(
        self,
        model: str = "Qwen2.5-VL-7B-Instruct",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        **kwargs
    ):
        self.model_name = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        logger.info(f"Initializing Qwen-VL API with model: {model}")

    def process_images(self, messages: List[Dict[str, Any]]) -> List[str]:
        """处理消息中的图片"""
        logger.info("Processing images from messages")
        images = []
        
        for message in messages:
            if message["role"] == "user":
                content = message["content"]
                
                # 处理列表形式的content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url")
                                if image_url:
                                    if image_url.startswith("data:image"):
                                        images.append(image_url)
                                    else:
                                        logger.warning(f"Unsupported image URL format: {image_url[:30]}...")
                            elif item.get("type") == "image":
                                image_data = item.get("image")
                                if isinstance(image_data, str):
                                    images.append(image_data)
                                elif isinstance(image_data, dict) and "url" in image_data:
                                    images.append(image_data["url"])
                
                # 处理字符串形式的content
                elif isinstance(content, str) and "data:image" in content:
                    # 尝试提取base64图片数据
                    try:
                        pattern = r"data:image\/[^;]+;base64,[^\"']+"
                        matches = re.findall(pattern, content)
                        images.extend(matches)
                    except Exception as e:
                        logger.error(f"Error extracting base64 image: {str(e)}")

        logger.info(f"Found {len(images)} images in total")
        return images

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str
    ) -> AsyncIterator[str]:
        """异步生成聊天回复"""
        try:
            logger.info(f"Starting chat completion")
            # 处理图片
            images = self.process_images(messages)
            if images:
                logger.info(f"Found {len(images)} images in messages")
            
            # 准备请求数据
            new_messages = []
            
            # 添加系统消息
            if system:
                new_messages.append({"role": "system", "content": system})
            
            # 处理用户消息
            for msg in messages:
                if msg["role"] == "user":
                    # 如果消息内容是列表（可能包含图片）
                    if isinstance(msg["content"], list):
                        new_content = []
                        for item in msg["content"]:
                            if item.get("type") == "text":
                                new_content.append({"type": "text", "text": item["text"]})
                            elif item.get("type") == "image_url":
                                url = item["image_url"].get("url")
                                if url:
                                    new_content.append({"type": "image", "image": url})
                            elif item.get("type") == "image":
                                new_content.append(item)
                        new_messages.append({"role": "user", "content": new_content})
                    else:
                        # 如果是纯文本消息
                        new_messages.append(msg)
                    
                else:
                    # 非用户消息直接添加
                    new_messages.append(msg)
            
            # 给最后一个消息的content添加 hello world 和字数限制
            if new_messages and isinstance(new_messages[-1], dict):
                last_msg = new_messages[-1]
                if isinstance(last_msg["content"], str):
                    last_msg["content"] = last_msg["content"] + "----你叫小绿龙，你很可爱，家住深圳南山区深圳大学。----，-----回答不要超过一句话----"
                elif isinstance(last_msg["content"], list):
                    # 如果是列表，找到最后一个文本元素并添加
                    for item in reversed(last_msg["content"]):
                        if isinstance(item, dict) and item.get("type") == "text":
                            item["text"] = item["text"] + "--------你叫小绿龙，你很可爱，家住深圳南山区深圳大学。----，-----回答不要超过一句话---"
                            break
            
            logger.info(f"==========new_message: {new_messages[-1]}")
            
            payload = {
                "model": self.model_name,
                "messages": new_messages,
                "temperature": self.temperature,
                "stream": True
            }
            
            logger.info(f"Sending request to {self.base_url}")
            logger.debug(f"Request payload: {payload}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        raise ValueError(f"API request failed: {error_text}")

                    logger.info(f"API request successful with status {response.status}")
                    # 处理流式响应
                    buffer = ""
                    response_started = False
                    async for line in response.content:
                        if line:
                            try:
                                line = line.decode('utf-8').strip()
                                if not response_started:
                                    logger.info("Started receiving response stream")
                                    response_started = True
                                logger.debug(f"Received line: {line}")
                                if line.startswith('data: '):
                                    line = line[6:]  # 移除 "data: " 前缀
                                if line:
                                    chunk = json.loads(line)
                                    logger.debug(f"Parsed chunk: {chunk}")
                                    if chunk.get("choices") and chunk["choices"][0].get("delta"):
                                        content = chunk["choices"][0]["delta"].get("content", "")
                                        if content:
                                            buffer += content
                                            logger.info(f"Yielding content: {content}")
                                            yield content
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON: {line}")
                            except Exception as e:
                                logger.error(f"Error processing response: {str(e)}")
                                raise

            if not response_started:
                logger.warning("No response stream was received")
            logger.info(f"Chat completion finished. Total response length: {len(buffer)}")

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
