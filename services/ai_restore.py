import os
import sys
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.getenv("REALPHOTO_AI_DIR", ROOT / "external_ai"))

def _torch_status():
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda else "CPU"
        return cuda, name
    except Exception:
        return False, "PyTorch unavailable"

def _find_repo(names):
    for name in names:
        p = EXTERNAL / name
        if p.exists():
            return p
    return None

def ai_status():
    cuda, device_name = _torch_status()
    return {
        "cuda": cuda,
        "device_name": device_name,
        "realesrgan": _find_repo(["Real-ESRGAN", "realesrgan"]) is not None,
        "gfpgan": _find_repo(["GFPGAN", "gfpgan"]) is not None,
        "codeformer": _find_repo(["CodeFormer", "codeformer"]) is not None,
        "swinir": _find_repo(["SwinIR", "swinir"]) is not None,
    }

def _run(cmd, cwd, timeout=900):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = proc.stdout[-5000:] if proc.stdout else "Unknown error"
        raise RuntimeError(tail)
    return proc.stdout

def _load_first_image(folder):
    candidates = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(glob.glob(str(Path(folder) / "**" / ext), recursive=True))
    if not candidates:
        return None
    # Most tools create more than one intermediate result; choose newest image.
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Image.open(candidates[0]).convert("RGB")

def run_ai_restore(image, module, scale=2, tile=400, fidelity=0.8, face_only=False):
    status = ai_status()
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="realphoto_ai_") as tmp:
        tmp = Path(tmp)
        inp = tmp / "input.png"
        out = tmp / "output"
        out.mkdir()
        image.save(inp)

        try:
            if module.startswith("Real-ESRGAN"):
                repo = _find_repo(["Real-ESRGAN", "realesrgan"])
                if repo is None:
                    return None, "Real-ESRGAN не установлен. Запустите scripts/setup_ai.*"
                cmd = [
                    sys.executable, "inference_realesrgan.py",
                    "-n", "RealESRGAN_x4plus",
                    "-i", str(inp),
                    "-o", str(out),
                    "--outscale", str(scale),
                    "--tile", str(tile),
                    "--ext", "png",
                ]
                _run(cmd, repo)
                result = _load_first_image(out)
                return result, "Real-ESRGAN выполнен."

            if module.startswith("GFPGAN"):
                repo = _find_repo(["GFPGAN", "gfpgan"])
                if repo is None:
                    return None, "GFPGAN не установлен. Запустите scripts/setup_ai.*"
                cmd = [
                    sys.executable, "inference_gfpgan.py",
                    "-i", str(inp),
                    "-o", str(out),
                    "-v", "1.3",
                    "-s", str(scale),
                    "-bg_tile", str(tile),
                    "-ext", "png",
                ]
                if face_only:
                    cmd += ["-bg_upsampler", "none"]
                else:
                    cmd += ["-bg_upsampler", "realesrgan"]
                _run(cmd, repo)
                result = _load_first_image(out)
                return result, "GFPGAN выполнен."

            if module.startswith("CodeFormer"):
                repo = _find_repo(["CodeFormer", "codeformer"])
                if repo is None:
                    return None, "CodeFormer не установлен. Запустите scripts/setup_ai.*"
                cmd = [
                    sys.executable, "inference_codeformer.py",
                    "-w", str(fidelity),
                    "--input_path", str(inp),
                    "--output_path", str(out),
                    "--upscale", str(scale),
                ]
                if not face_only:
                    cmd += ["--bg_upsampler", "realesrgan"]
                _run(cmd, repo)
                result = _load_first_image(out)
                return result, f"CodeFormer выполнен с fidelity={fidelity:.2f}."

            if module.startswith("SwinIR"):
                repo = _find_repo(["SwinIR", "swinir"])
                if repo is None:
                    return None, "SwinIR не установлен. Запустите scripts/setup_ai.*"
                # SwinIR expects a folder.
                in_dir = tmp / "swin_input"
                in_dir.mkdir()
                shutil.copy2(inp, in_dir / "input.png")
                cmd = [
                    sys.executable, "main_test_swinir.py",
                    "--task", "real_sr",
                    "--scale", str(scale),
                    "--folder_lq", str(in_dir),
                ]
                _run(cmd, repo)
                # Official script writes under results/swinir_real_sr_x*
                result = _load_first_image(repo / "results")
                if result is None:
                    result = _load_first_image(out)
                return result, "SwinIR выполнен."

            return None, "Неизвестный AI-модуль."

        except subprocess.TimeoutExpired:
            return None, "AI-модуль превысил лимит времени. Уменьшите изображение или tile."
        except Exception as e:
            return None, "Ошибка AI backend: " + str(e)[-3500:]
