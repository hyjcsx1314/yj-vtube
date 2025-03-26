import os
import base64
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import logging
from PIL import Image
import io
import time
import json
import asyncio

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ResponseFormat(BaseModel):
    type: str = "text"

class Parameters(BaseModel):
    temperature: float = Field(default=0.7, ge=0, lt=2)
    top_p: float = Field(default=0.8, gt=0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    response_format: ResponseFormat = Field(default=ResponseFormat(type="text"))
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: int = Field(default=512, gt=0)

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "qwen2.5-vl"
    parameters: Optional[Parameters] = Parameters()

class ChatResponse(BaseModel):
    output: Dict
    usage: Dict

def decode_base64_image(base64_string: str) -> Image.Image:
    """将base64字符串解码为PIL Image对象"""
    try:
        # 如果base64字符串包含header，去掉header
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # 解码base64字符串
        image_data = base64.b64decode(base64_string)
        
        # 将二进制数据转换为PIL Image对象
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logging.error(f"Failed to decode base64 image: {str(e)}")
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

class QwenVLServer:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        self.app = FastAPI(title="Qwen2.5-VL API Server")
        self.model = None
        self.processor = None
        self.model_path = model_path
        self.device = device
        
        # 设置日志
        self.logger = logging.getLogger("QwenVLServer")
        
        # 添加CORS支持
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.setup_routes()
        self.logger.info(f"Server initialized with model path: {model_path}, device: {device}")

    def load_model(self):
        """加载模型和处理器"""
        try:
            if self.model is None:
                start_time = time.time()
                self.logger.info(f"Loading model from {self.model_path}")
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True
                )
                self.logger.info("Model loaded successfully")
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                self.logger.info("Processor loaded successfully")
                
                load_time = time.time() - start_time
                self.logger.info(f"Total model loading time: {load_time:.2f} seconds")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    def process_images(self, messages: List[Message]) -> List[Image.Image]:
        """处理消息中的图片"""
        images = []
        has_image_content = False
        
        for msg in messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == "image" and item.image:
                        has_image_content = True
                        try:
                            self.logger.info(f"Processing image data of length: {len(item.image) if item.image else 0}")
                            image = decode_base64_image(item.image)
                            self.logger.info(f"Successfully decoded image, size: {image.size}, mode: {image.mode}")
                            
                            # 确保图片格式正确
                            if image.mode not in ("RGB", "RGBA"):
                                image = image.convert("RGB")
                                self.logger.info("Converted image to RGB mode")
                            
                            images.append(image)
                        except Exception as e:
                            self.logger.error(f"Error processing image: {str(e)}", exc_info=True)
                            raise HTTPException(
                                status_code=400,
                                detail=f"Invalid image data: {str(e)}"
                            )
            elif isinstance(msg.content, str) and "data:image" in msg.content:
                has_image_content = True
                try:
                    self.logger.info("Found inline image in text content")
                    image = decode_base64_image(msg.content)
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")
                    images.append(image)
                except Exception as e:
                    self.logger.error(f"Error processing inline image: {str(e)}", exc_info=True)
        
        if not has_image_content:
            self.logger.info("No image content found in messages")
            return []
                    
        self.logger.info(f"Processed {len(images)} images successfully")
        return images

    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatRequest):
            try:
                request_start_time = time.time()
                self.logger.info(f"Received chat request for model: {request.model}")
                self.logger.debug(f"Request parameters: {request.parameters}")
                
                # 确保模型已加载
                if self.model is None:
                    self.load_model()
                
                # 处理消息格式
                processed_messages = []
                # 只处理最后一轮对话
                last_user_msg_idx = -1
                for i in range(len(request.messages)-1, -1, -1):
                    if request.messages[i].role == "user":
                        last_user_msg_idx = i
                        break
                
                if last_user_msg_idx >= 0:
                    # 只取最后一个用户消息和可能的助手回复
                    relevant_messages = request.messages[last_user_msg_idx:]
                    self.logger.info(f"Processing last round messages, count: {len(relevant_messages)}")
                else:
                    relevant_messages = request.messages[-1:]
                    self.logger.info("No user message found, using last message only")

                for msg in relevant_messages:
                    if isinstance(msg.content, str):
                        processed_messages.append({
                            "role": msg.role,
                            "content": [{"type": "text", "text": msg.content}]
                        })
                    else:
                        processed_messages.append({
                            "role": msg.role,
                            "content": [
                                item.model_dump(exclude_none=True) 
                                for item in msg.content
                            ]
                        })
                
                self.logger.debug(f"Processed messages: {processed_messages}")
                
                # 处理图片 - 只处理当前消息中的图片
                images = self.process_images(relevant_messages)
                
                # 应用chat模板
                try:
                    text = self.processor.apply_chat_template(
                        processed_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    self.logger.debug(f"Applied chat template, text length: {len(text)}")
                    self.logger.debug(f"Chat template text: {text[:200]}...")  # 只记录前200个字符
                except Exception as e:
                    self.logger.error(f"Error applying chat template: {str(e)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error applying chat template: {str(e)}"
                    )
                
                # 准备输入
                try:
                    inputs = self.processor(
                        text=[text],
                        images=images if images else None,  # 确保当images为空列表时传入None
                        return_tensors="pt"
                    ).to(self.model.device)
                    self.logger.info(f"Inputs prepared, shape: {inputs.input_ids.shape}, has_images: {bool(images)}")
                except Exception as e:
                    self.logger.error(f"Error processing inputs: {str(e)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing inputs: {str(e)}"
                    )

                # 使用批量生成和分块返回的方式
                async def generate_stream():
                    try:
                        self.logger.info("Starting text generation")
                        # 生成完整响应
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=request.parameters.max_tokens,
                            temperature=request.parameters.temperature,
                            top_p=request.parameters.top_p,
                            repetition_penalty=1.1,
                            do_sample=True,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id
                        )
                        self.logger.info("Text generation completed")

                        # 解码生成的文本
                        generated_text = self.processor.batch_decode(
                            outputs[:, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True
                        )[0]

                        self.logger.info(f"Generated text length: {len(generated_text)}")
                        self.logger.debug(f"Generated text: {generated_text}")

                        # 按块发送流式响应
                        chunk_size = 10  # 每个块的字符数
                        total_chunks = (len(generated_text) + chunk_size - 1) // chunk_size
                        chunks_sent = 0
                        
                        for i in range(0, len(generated_text), chunk_size):
                            chunks_sent += 1
                            chunk = generated_text[i:i+chunk_size]
                            response_chunk = {
                                "id": f"chatcmpl-{int(time.time()*1000)}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": chunk
                                    },
                                    "finish_reason": None
                                }]
                            }
                            chunk_data = f"data: {json.dumps(response_chunk)}\n\n"
                            self.logger.debug(f"Sending chunk {chunks_sent}/{total_chunks}: {chunk_data}")
                            yield chunk_data
                            await asyncio.sleep(0.01)  # 控制发送速率

                        # 发送结束信号
                        final_chunk = {
                            "id": f"chatcmpl-{int(time.time()*1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        self.logger.info("Sending final chunk")
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        self.logger.info("Stream completed successfully")
                    except Exception as e:
                        self.logger.error(f"Error in generate_stream: {str(e)}", exc_info=True)
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error in generate_stream: {str(e)}"
                        )

                # 返回流式响应
                self.logger.info("Starting streaming response")
                response = StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream",
                        "X-Accel-Buffering": "no"
                    }
                )
                self.logger.info("StreamingResponse object created successfully")
                return response

            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        self.logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = QwenVLServer(model_path=args.model_path, device=args.device)
    server.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
