import requests
import os
import argparse
from pathlib import Path

def test_voice_clone(
    api_url: str,
    audio_path: str,
    text: str,
    prompt_text: str = None,
    output_dir: str = "test_outputs"
):
    """
    Test the voice cloning API endpoint.
    
    Args:
        api_url: The base URL of the API
        audio_path: Path to the reference audio file
        text: Text to be converted to speech
        prompt_text: Text corresponding to the reference audio (optional)
        output_dir: Directory to save the output audio
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the API endpoint URL
    url = f"{api_url.rstrip('/')}/clone_voice"
    
    # Prepare the files and data for the request
    files = {
        'prompt_audio': ('reference.wav', open(audio_path, 'rb'), 'audio/wav'),
    }
    
    data = {
        'text': text,
    }
    
    if prompt_text:
        data['prompt_text'] = prompt_text
    
    try:
        print(f"Sending request to {url}")
        print(f"Text to convert: {text}")
        if prompt_text:
            print(f"Prompt text: {prompt_text}")
        print(f"Using reference audio: {audio_path}")
        
        # Send POST request to the API
        response = requests.post(url, files=files, data=data)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Save the response content as an audio file
        output_path = os.path.join(output_dir, "cloned_output.wav")
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"\nSuccess! Generated audio saved to: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError occurred while making the request: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")
        return False
    
    finally:
        # Close the file
        files['prompt_audio'][1].close()

def main():
    parser = argparse.ArgumentParser(description="Test Spark-TTS Voice Cloning API")
    parser.add_argument("--url", default="http://192.168.1.9:8000",
                      help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--audio", default="全民制作人们大家好，我是练习时长两年半的个人练习生蔡徐坤。.wav",
                      help="Path to reference audio file (WAV format)")
    parser.add_argument("--text", default="我中午想吃火锅",
                      help="Text to be converted to speech")
    parser.add_argument("--prompt-text",default="全民制作人们大家好，我是练习时长两年半的个人练习生蔡徐坤",
                      help="Text corresponding to the reference audio (optional)")
    parser.add_argument("--output-dir", default="test_outputs",
                      help="Directory to save output (default: test_outputs)")
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    # Run the test
    test_voice_clone(
        api_url=args.url,
        audio_path=args.audio,
        text=args.text,
        prompt_text=args.prompt_text,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 