import argparse
import asyncio
import io
import os
import sys

from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import Optional
import numpy as np
import torch
import librosa
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import torchaudio


# 定义音频请求的队列
audio_request_queue = asyncio.Queue()

# FastAPI实例
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许的HTTP方法
    allow_headers=["*"],  # 允许的HTTP头部
)

# 读取模组路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

# 预定义变量
max_val = 0.8

class AudioRequest(BaseModel):
    tts_text: str
    mode: str
    sft_dropdown: Optional[str] = None
    prompt_text: Optional[str] = None
    instruct_text: Optional[str] = None
    seed: Optional[int] = 0
    stream: Optional[bool] = False
    speed: Optional[float] = 1.0
    prompt_voice: Optional[str] = None

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

# 用于存储活跃的 WebSocket 连接
active_connections = set()

async def inference_audio(request: AudioRequest):
    set_all_random_seed(request.seed)
    prompt_speech_16k = load_wav(request.prompt_voice, 16000) if request.prompt_voice else None

    if request.mode == 'zero_shot':
        return await asyncio.to_thread(cosyvoice.inference_zero_shot, request.tts_text, request.prompt_text, prompt_speech_16k, stream=request.stream, speed=request.speed)
    elif request.mode == 'instruct':
        return await asyncio.to_thread(cosyvoice.inference_instruct2, request.tts_text, request.instruct_text, prompt_speech_16k, stream=request.stream, speed=request.speed)
    elif request.mode == 'sft':
        return await asyncio.to_thread(cosyvoice.inference_sft, request.tts_text, request.sft_dropdown, stream=request.stream, speed=request.speed)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

# 音频生成函数（流式输出）
async def generate_audio_stream(request: AudioRequest):
    result = await inference_audio(request)

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    # 流式输出
    for i in result:
        audio_data = i['tts_speech'].numpy().flatten()
        audio_bytes = (audio_data * (2**15)).astype(np.int16).tobytes()
        yield audio_bytes

# 音频生成函数（非流式输出）
async def generate_audio_buffer(request: AudioRequest):
    result = await inference_audio(request)

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    # 非流式输出
    buffer = io.BytesIO()
    tts_speeches = [j['tts_speech'] for i, j in enumerate(result)]
    audio_data = torch.concat(tts_speeches, dim=1)
    torchaudio.save(buffer, audio_data, cosyvoice.sample_rate, format="wav")
    buffer.seek(0)
    return buffer

@app.post("/text-tts")
async def text_tts(request: AudioRequest):
    if not request.tts_text:
        raise HTTPException(status_code=400, detail="Query parameter 'tts_text' is required")
    
    if request.stream:
        # 流式输出
        return StreamingResponse(generate_audio_stream(request), media_type="audio/pcm")
    else:
        # 非流式输出
        buffer = await generate_audio_buffer(request)
        return Response(buffer.read(), media_type="audio/wav")

@app.post("/push-text-tts")
async def push_text_tts(request: AudioRequest, background_tasks: BackgroundTasks):
    if not request.tts_text:
        raise HTTPException(status_code=400, detail="Query parameter 'tts_text' is required")

    await audio_request_queue.put(request)  # 将请求放入异步队列

    background_tasks.add_task(process_audio_requests)

    return Response(content="Request received", media_type="text/plain")

async def process_audio_requests():
    while not audio_request_queue.empty():
        request = await audio_request_queue.get()  # 从异步队列获取请求
        if active_connections:
            print(f"活跃连接数量: {len(active_connections)}")
            await send_audio_to_connections(request)
        await asyncio.sleep(0.2)  # 控制处理速度

async def send_audio_to_connections(request: AudioRequest):
    # 创建 active_connections 的副本进行迭代
    connections_copy = list(active_connections)
    for connection in connections_copy:
        try:
            async for audio_bytes in generate_audio_stream(request):
                await connection.send_bytes(audio_bytes)
        except Exception as e:
            print(f"在处理连接时发生错误: {e}")
            active_connections.discard(connection)  # 使用 discard 而不是 remove

@app.websocket("/ws/monitor-audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 确保在接收请求数据前调用 accept
    active_connections.add(websocket)  # 添加新连接到活跃列表
    try:
        while True:
            # 可选择在这里接收来自客户端的消息
            # 可以设置一个条件，以便在特定情况下退出循环
            data = await websocket.receive_text()
            print(f"Received message: {data}")

    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        active_connections.remove(websocket)
        await websocket.close()

# 获取队列状态
@app.get("/queue-status")
async def get_queue_status():
    queue_size = audio_request_queue.qsize()
    active_connections_count = len(active_connections)
    return JSONResponse(content={
        "queue_size": queue_size,
        "active_connections": active_connections_count
    })

# 获音色列表
@app.get("/sft_spk")
async def get_sft_spk():
    sft_spk = cosyvoice.list_available_spks()  # 获取音色列表
    return JSONResponse(content=sft_spk)  # 返回 JSON 格式的响应


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', help='local path or modelscope repo id')
    args = parser.parse_args()
    # 初始化CosyVoice模型

    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=False) 
    
    # 启动FastAPI
    uvicorn.run(app, host='0.0.0.0', port=50000)

