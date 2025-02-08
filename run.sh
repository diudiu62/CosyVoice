#!/bin/bash

# 设置conda环境
#conda activate /root/miniconda3/envs/cosyvoice
# source环境
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# 激活虚拟环境
conda activate cosyvoice

# 切换工作目录
cd /home/CosyVoice

export PYTHONPATH=third_party/Matcha-TTS
export CUDA_VISIBLE_DEVICES=1

# 默认模型路径
model_dir="pretrained_models/CosyVoice2-0.5B"

# 处理参数
while getopts ":HSI" opt; do
  case ${opt} in
    H)
      model_dir="pretrained_models/CosyVoice-300M-25Hz"
      ;;
    S)
      model_dir="pretrained_models/CosyVoice-300M-SFT"
      ;;
    I)
      model_dir="pretrained_models/CosyVoice-300M-Instruct"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done


# 启动应用
# CUDA_VISIBLE_DEVICES=1 nohup python3 webui.py --port 6006 --model_dir "$model_dir" > output.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup  python3 api.py --model_dir "$model_dir" > output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup fastapi run --port 8000 > output.log 2>&1 &