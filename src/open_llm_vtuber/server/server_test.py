import requests
import base64
import json
import os
from PIL import Image
import io
import sseclient
import logging
import traceback

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def encode_image(image_path):
    """将图片文件编码为base64字符串"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        # 读取图片文件并转换为base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

def handle_stream_response(response):
    """处理流式响应的通用函数"""
    logger.info("开始处理流式响应")
    logger.debug(f"响应头: {dict(response.headers)}")
    
    try:
        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                logger.debug(f"收到数据行: {line}")
                if line.startswith('data: '):
                    data_str = line[6:]  # 去掉 'data: ' 前缀
                    if data_str == '[DONE]':
                        logger.info("收到流式响应结束标记")
                        break
                        
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                full_text += content
                                print(content, end="", flush=True)
                                logger.debug(f"处理内容片段: {content}")
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON解析错误: {je}")
                        logger.error(f"问题数据: {data_str}")
                    except Exception as e:
                        logger.error(f"处理事件时发生错误: {str(e)}")
                        logger.error(traceback.format_exc())
                        
        logger.info("流式响应处理完成")
        print("\n完整响应文本:", full_text)
        return full_text
        
    except Exception as e:
        logger.error(f"处理流式响应时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def test_qwen_api(stream=False):
    url = "http://127.0.0.1:11433/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # 先测试纯文本
    text_payload = {
        "messages": [
            {
                "role": "user",
                "content": "你好，请自我介绍一下"
            }
        ],
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 512
        }
    }

    try:
        logger.info("开始纯文本请求测试")
        logger.debug(f"请求数据: {json.dumps(text_payload, ensure_ascii=False)}")
        
        if stream:
            logger.info("使用流式响应模式")
            response = requests.post(url, json=text_payload, headers=headers, stream=True)
            logger.info(f"状态码: {response.status_code}")
            logger.debug(f"响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                handle_stream_response(response)
            else:
                logger.error(f"请求失败，状态码: {response.status_code}")
                logger.error(f"错误响应: {response.text}")
                response.raise_for_status()
        else:
            logger.info("使用普通响应模式")
            response = requests.post(url, json=text_payload, headers=headers)
            logger.info(f"状态码: {response.status_code}")
            logger.debug(f"响应头: {dict(response.headers)}")
            logger.debug(f"原始响应: {response.text}")
            
            response.raise_for_status()
            logger.info("纯文本测试成功!")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            
    except requests.exceptions.RequestException as e:
        logger.error(f"纯文本测试失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"响应内容: {e.response.text}")
        return
    except Exception as e:
        logger.error(f"纯文本测试中发生意外错误: {str(e)}")
        logger.error(traceback.format_exc())
        return

    # 测试图片+文本
    try:
        image_path = r"C:\Users\Administrator\Pictures\Screenshots\2025.png"  # 确保这个文件存在
        logger.info(f"开始图片+文本请求测试，图片路径: {image_path}")
        
        image_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "描述这张图片"
                        },
                        {
                            "type": "image",
                            "image": encode_image(image_path)
                        }
                    ]
                }
            ],
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 512
            }
        }

        logger.debug("图片请求数据 (不包含图片数据): " + json.dumps({
            **image_payload,
            'messages': [{
                **image_payload['messages'][0],
                'content': [
                    {'type': 'text', 'text': '描述这张图片'},
                    {'type': 'image', 'image': '[BASE64_IMAGE_DATA]'}
                ]
            }]
        }, ensure_ascii=False))

        if stream:
            logger.info("使用流式响应模式处理图片请求")
            response = requests.post(url, json=image_payload, headers=headers, stream=True)
            logger.info(f"状态码: {response.status_code}")
            logger.debug(f"响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                handle_stream_response(response)
            else:
                logger.error(f"图片请求失败，状态码: {response.status_code}")
                logger.error(f"错误响应: {response.text}")
                response.raise_for_status()
        else:
            logger.info("使用普通响应模式处理图片请求")
            response = requests.post(url, json=image_payload, headers=headers)
            logger.info(f"状态码: {response.status_code}")
            logger.debug(f"响应头: {dict(response.headers)}")
            logger.debug(f"原始响应: {response.text}")
            
            response.raise_for_status()
            logger.info("图片+文本测试成功!")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            
    except FileNotFoundError as e:
        logger.error(f"图片文件错误: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"图片+文本测试失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"响应内容: {e.response.text}")
    except Exception as e:
        logger.error(f"图片+文本测试中发生意外错误: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 默认使用流式响应
    test_qwen_api(stream=True)
    
    # 如果需要测试非流式响应，可以取消下面的注释
    # test_qwen_api(stream=False)
