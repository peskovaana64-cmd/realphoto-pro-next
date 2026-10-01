$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path external_ai | Out-Null
Set-Location external_ai

if (!(Test-Path "Real-ESRGAN")) {
    git clone https://github.com/xinntao/Real-ESRGAN.git
}
python -m pip install -r Real-ESRGAN\requirements.txt
python -m pip install -e Real-ESRGAN

if (!(Test-Path "GFPGAN")) {
    git clone https://github.com/TencentARC/GFPGAN.git
}
python -m pip install -r GFPGAN\requirements.txt
python -m pip install -e GFPGAN

if (!(Test-Path "CodeFormer")) {
    git clone https://github.com/sczhou/CodeFormer.git
}
python -m pip install -r CodeFormer\requirements.txt
Push-Location CodeFormer
python basicsr\setup.py develop
Pop-Location

if (!(Test-Path "SwinIR")) {
    git clone https://github.com/JingyunLiang/SwinIR.git
}

Write-Host ""
Write-Host "AI repositories installed."
Write-Host "Weights are downloaded by official scripts on first use where supported."
