#!/usr/bin/env bash
set -e

mkdir -p external_ai
cd external_ai

if [ ! -d "Real-ESRGAN" ]; then
  git clone https://github.com/xinntao/Real-ESRGAN.git
fi
python -m pip install -r Real-ESRGAN/requirements.txt
python -m pip install -e Real-ESRGAN

if [ ! -d "GFPGAN" ]; then
  git clone https://github.com/TencentARC/GFPGAN.git
fi
python -m pip install -r GFPGAN/requirements.txt
python -m pip install -e GFPGAN

if [ ! -d "CodeFormer" ]; then
  git clone https://github.com/sczhou/CodeFormer.git
fi
python -m pip install -r CodeFormer/requirements.txt
python CodeFormer/basicsr/setup.py develop || true

if [ ! -d "SwinIR" ]; then
  git clone https://github.com/JingyunLiang/SwinIR.git
fi

echo ""
echo "AI repositories installed."
echo "Weights are downloaded by the official scripts on first use where supported."
