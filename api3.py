import time
import io, os, sys
from flask_cors import CORS
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import numpy as np
from flask import Flask, request, Response
import torch
import torchaudio
import librosa

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B',load_jit=True, load_onnx=True, load_trt=False)

print(cosyvoice.list_avaliable_spks())

app = Flask(__name__)
CORS(app)  

max_val = 0.8

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

@app.route("/api", methods=['POST'])
def stream():
    question_data = request.get_json()
    tts_text = question_data.get('query')
    new_dropdown = question_data.get('new_dropdown')
    instruct_text = question_data.get('instruct_text')
    stream = question_data.get('stream')
    prompt_speech_16k = postprocess(load_wav('example.wav',16000))

    if not tts_text:
        return {"error": "Query parameter 'query' is required"}, 400

    def generate_stream():
        
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream,new_dropdown=new_dropdown):
            audio_data = i['tts_speech'].numpy().flatten()
            yield audio_data.tobytes()

    return Response(generate_stream(), mimetype="audio/pcm") 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=50000)
