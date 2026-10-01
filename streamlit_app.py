import io
import math
import os
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from services.ai_restore import ai_status, run_ai_restore
from services.external_providers import external_provider_status, run_external_provider

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

st.set_page_config(
    page_title="REALPHOTO PRO 2.0",
    page_icon="📷",
    layout="wide",
)

st.title("REALPHOTO PRO 2.0")
st.caption("Render → Real Photograph · бесплатный CPU-пайплайн + подготовка к AI/GPU-модулям")

PRESETS = {
    "Архитектура / ЖК": {
        "strength": 0.42, "local_contrast": 0.28, "texture": 0.20,
        "shadows": 0.10, "highlights": -0.12, "warmth": 0.02,
        "saturation": -0.03, "denoise": 0.08, "sharp": 0.18,
        "grain": 0.05, "bloom": 0.00,
    },
    "Интерьер": {
        "strength": 0.36, "local_contrast": 0.22, "texture": 0.16,
        "shadows": 0.14, "highlights": -0.10, "warmth": 0.05,
        "saturation": -0.02, "denoise": 0.10, "sharp": 0.14,
        "grain": 0.04, "bloom": 0.01,
    },
    "Люди / lifestyle": {
        "strength": 0.30, "local_contrast": 0.16, "texture": 0.12,
        "shadows": 0.10, "highlights": -0.08, "warmth": 0.04,
        "saturation": -0.02, "denoise": 0.12, "sharp": 0.10,
        "grain": 0.04, "bloom": 0.01,
    },
    "Ночной / синий час": {
        "strength": 0.38, "local_contrast": 0.25, "texture": 0.16,
        "shadows": 0.18, "highlights": -0.20, "warmth": -0.02,
        "saturation": -0.05, "denoise": 0.16, "sharp": 0.12,
        "grain": 0.06, "bloom": 0.02,
    },
    "Нейтральный": {
        "strength": 0.25, "local_contrast": 0.15, "texture": 0.10,
        "shadows": 0.05, "highlights": -0.05, "warmth": 0.00,
        "saturation": 0.00, "denoise": 0.05, "sharp": 0.10,
        "grain": 0.03, "bloom": 0.00,
    },
}

def pil_to_np(image):
    return np.asarray(image.convert("RGB")).astype(np.uint8)

def np_to_pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

def resize_long_edge(image, target):
    if target == "Оригинал":
        return image.copy()
    px = {"2K": 2048, "4K": 3840}[target]
    w, h = image.size
    scale = px / max(w, h)
    if abs(scale - 1.0) < 0.01:
        return image.copy()
    return image.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.Resampling.LANCZOS)

def protect_highlights(arr, amount):
    if amount <= 0:
        return arr
    x = arr.astype(np.float32) / 255.0
    lum = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    mask = np.clip((lum - 0.55) / 0.45, 0, 1)[..., None]
    compressed = x / (1.0 + amount * 0.45 * x)
    y = x*(1-mask*amount) + compressed*(mask*amount)
    return np.clip(y*255, 0, 255).astype(np.uint8)

def lift_shadows(arr, amount):
    if abs(amount) < 1e-4:
        return arr
    x = arr.astype(np.float32) / 255.0
    lum = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    mask = np.clip((0.60-lum)/0.60, 0, 1)[..., None]
    if amount > 0:
        y = x + mask * amount * 0.18 * (1-x)
    else:
        y = x + mask * amount * 0.12 * x
    return np.clip(y*255, 0, 255).astype(np.uint8)

def adjust_temperature(arr, warmth, tint):
    x = arr.astype(np.float32)
    # warmth: warm => red up, blue down
    x[...,0] *= 1.0 + warmth*0.13
    x[...,2] *= 1.0 - warmth*0.13
    # tint: magenta => red/blue up, green down
    x[...,0] *= 1.0 + tint*0.05
    x[...,2] *= 1.0 + tint*0.05
    x[...,1] *= 1.0 - tint*0.08
    return np.clip(x, 0, 255).astype(np.uint8)

def cv_local_contrast(arr, strength):
    if not CV2_AVAILABLE or strength <= 0:
        return arr
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clip = 1.2 + strength * 1.8
    grid = 8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    enhanced = clahe.apply(l)
    l2 = cv2.addWeighted(l, 1.0-strength*0.55, enhanced, strength*0.55, 0)
    return cv2.cvtColor(cv2.merge((l2,a,b)), cv2.COLOR_LAB2RGB)

def cv_denoise(arr, strength):
    if not CV2_AVAILABLE or strength <= 0:
        return arr
    h = 2 + strength*8
    return cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)

def add_texture(arr, amount):
    if amount <= 0:
        return arr
    # subtle high-frequency texture; avoids "plastic" smoothness
    img = np_to_pil(arr)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    hi = np.asarray(img).astype(np.float32) - np.asarray(blurred).astype(np.float32)
    out = np.asarray(img).astype(np.float32) + hi * (0.30 + amount*0.70)
    return np.clip(out,0,255).astype(np.uint8)

def add_grain(arr, amount, seed=13):
    if amount <= 0:
        return arr
    rng = np.random.default_rng(seed)
    h, w = arr.shape[:2]
    noise = rng.normal(0, 1.5 + amount*7.0, (h,w,1)).astype(np.float32)
    x = arr.astype(np.float32)
    # mostly luminance grain
    return np.clip(x + noise, 0, 255).astype(np.uint8)

def add_bloom(arr, amount):
    if amount <= 0 or not CV2_AVAILABLE:
        return arr
    x = arr.astype(np.float32) / 255.0
    lum = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    mask = np.clip((lum-0.70)/0.30, 0, 1)[...,None]
    glow = cv2.GaussianBlur((x*mask).astype(np.float32), (0,0), sigmaX=4.0)
    y = x + glow * amount * 0.35
    return np.clip(y*255,0,255).astype(np.uint8)


def _soft_mask_from_hsv(arr, kind):
    if not CV2_AVAILABLE:
        return np.zeros(arr.shape[:2], dtype=np.float32)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[...,0]
    s = hsv[...,1] / 255.0
    v = hsv[...,2] / 255.0

    if kind == "glass":
        # cool / low-sat bright zones, typical windows and reflections
        mask = ((s < 0.35) & (v > 0.28)).astype(np.float32)
        cool = (((h > 80) & (h < 125)) | (s < 0.12)).astype(np.float32)
        mask *= 0.55 + 0.45*cool

    elif kind == "greenery":
        mask = ((h > 28) & (h < 95) & (s > 0.20)).astype(np.float32)

    elif kind == "wood":
        warm = ((h < 28) | (h > 170)).astype(np.float32)
        mask = warm * (s > 0.18).astype(np.float32) * (v > 0.15).astype(np.float32)

    elif kind == "skin":
        # intentionally conservative; avoids aggressive face recoloring
        warm = (((h < 18) | (h > 170)) & (s > 0.12) & (s < 0.72) & (v > 0.28)).astype(np.float32)
        mask = warm

    elif kind == "metal":
        mask = ((s < 0.22) & (v > 0.18)).astype(np.float32)

    elif kind == "facade":
        # broad architectural midtone mask; intentionally excludes darkest and brightest areas
        mask = ((v > 0.15) & (v < 0.92)).astype(np.float32)

    else:
        mask = np.zeros(arr.shape[:2], dtype=np.float32)

    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3.0)
    return np.clip(mask, 0, 1)

def _blend_by_mask(base, edited, mask, opacity):
    if opacity <= 0:
        return base
    m = (mask * opacity)[...,None].astype(np.float32)
    out = base.astype(np.float32)*(1-m) + edited.astype(np.float32)*m
    return np.clip(out,0,255).astype(np.uint8)

def material_pass(arr, glass, facade, metal, wood, greenery, skin):
    if not CV2_AVAILABLE:
        return arr

    base = arr.copy()

    # Glass: add local clarity + slightly cooler, more reflective tonal separation.
    if glass > 0:
        g = cv_local_contrast(base, min(1.0, 0.35 + glass*0.50))
        gx = g.astype(np.float32)
        gx[...,2] *= 1.0 + glass*0.045
        gx[...,0] *= 1.0 - glass*0.018
        g = np.clip(gx,0,255).astype(np.uint8)
        base = _blend_by_mask(base, g, _soft_mask_from_hsv(base, "glass"), glass*0.55)

    # Facade: micro-contrast and anti-plastic surface structure.
    if facade > 0:
        f = cv_local_contrast(base, min(1.0, facade*0.70))
        f = add_texture(f, facade*0.65)
        base = _blend_by_mask(base, f, _soft_mask_from_hsv(base, "facade"), facade*0.48)

    # Metal: restrained crispness and highlight definition.
    if metal > 0:
        m = np_to_pil(base).filter(ImageFilter.UnsharpMask(
            radius=0.8, percent=int(25 + 70*metal), threshold=5
        ))
        m = np.asarray(m)
        base = _blend_by_mask(base, m, _soft_mask_from_hsv(base, "metal"), metal*0.45)

    # Wood: slightly warmer local midtones and tactile detail.
    if wood > 0:
        w = add_texture(base, wood*0.50).astype(np.float32)
        w[...,0] *= 1.0 + wood*0.035
        w[...,1] *= 1.0 + wood*0.012
        w = np.clip(w,0,255).astype(np.uint8)
        base = _blend_by_mask(base, w, _soft_mask_from_hsv(base, "wood"), wood*0.45)

    # Greenery: reduce neon CGI greens, add natural tonal variation.
    if greenery > 0:
        hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,1] *= 1.0 - greenery*0.10
        hsv[...,2] *= 1.0 - greenery*0.015
        gr = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        gr = cv_local_contrast(gr, greenery*0.35)
        base = _blend_by_mask(base, gr, _soft_mask_from_hsv(base, "greenery"), greenery*0.65)

    # Skin: conservative texture recovery, never aggressive sharpening.
    if skin > 0:
        s = add_texture(base, skin*0.18)
        s = cv_denoise(s, skin*0.06)
        base = _blend_by_mask(base, s, _soft_mask_from_hsv(base, "skin"), skin*0.22)

    return base


def apply_highlight_rolloff(arr, amount):
    if amount <= 0:
        return arr
    x = arr.astype(np.float32) / 255.0
    lum = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    mask = np.clip((lum - 0.58) / 0.42, 0, 1)[...,None]
    compressed = 1.0 - np.exp(-x * (1.55 - amount*0.30))
    y = x*(1-mask*amount*0.75) + compressed*(mask*amount*0.75)
    return np.clip(y*255,0,255).astype(np.uint8)

def lens_softness(arr, amount):
    if amount <= 0:
        return arr
    img = np_to_pil(arr)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=0.35 + amount*0.80))).astype(np.float32)
    base = arr.astype(np.float32)
    y = base*(1-amount*0.22) + blur*(amount*0.22)
    return np.clip(y,0,255).astype(np.uint8)

def halation_pass(arr, amount):
    if amount <= 0 or not CV2_AVAILABLE:
        return arr
    x = arr.astype(np.float32) / 255.0
    lum = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    mask = np.clip((lum-0.72)/0.28, 0, 1)[...,None]
    red = np.zeros_like(x)
    red[...,0] = x[...,0]
    red[...,1] = x[...,1]*0.28
    red[...,2] = x[...,2]*0.10
    glow = cv2.GaussianBlur((red*mask).astype(np.float32), (0,0), sigmaX=3.0+amount*6.0)
    y = x + glow*amount*0.20
    return np.clip(y*255,0,255).astype(np.uint8)

def chromatic_aberration(arr, amount):
    if amount <= 0 or not CV2_AVAILABLE:
        return arr
    px = max(1, int(round(amount*2.5)))
    out = arr.copy()
    out[...,0] = np.roll(arr[...,0], px, axis=1)
    out[...,2] = np.roll(arr[...,2], -px, axis=1)
    # blend conservatively
    return np.clip(arr.astype(np.float32)*0.82 + out.astype(np.float32)*0.18, 0,255).astype(np.uint8)

def atmospheric_depth(arr, amount):
    if amount <= 0:
        return arr
    h, w = arr.shape[:2]
    y = np.linspace(0,1,h, dtype=np.float32)[:,None]
    # more haze toward upper/far part of frame
    mask = np.clip((1.0-y)*amount*0.18, 0, 0.18)[...,None]
    haze = np.full_like(arr, 235, dtype=np.float32)
    out = arr.astype(np.float32)*(1-mask) + haze*mask
    return np.clip(out,0,255).astype(np.uint8)

def sensor_noise(arr, amount, seed=37):
    if amount <= 0:
        return arr
    rng = np.random.default_rng(seed)
    h,w = arr.shape[:2]
    # chroma + luminance, subtle
    lum = rng.normal(0, 0.8 + amount*3.0, (h,w,1))
    chroma = rng.normal(0, 0.25 + amount*0.9, (h,w,3))
    out = arr.astype(np.float32) + lum + chroma
    return np.clip(out,0,255).astype(np.uint8)

def apply_lens_profile(arr, lens_name, lens_strength):
    if lens_strength <= 0:
        return arr

    if lens_name == "35mm Documentary":
        arr = lens_softness(arr, lens_strength*0.22)
        arr = chromatic_aberration(arr, lens_strength*0.12)
        arr = atmospheric_depth(arr, lens_strength*0.06)

    elif lens_name == "50mm Natural":
        arr = lens_softness(arr, lens_strength*0.14)
        arr = chromatic_aberration(arr, lens_strength*0.06)

    elif lens_name == "85mm Premium":
        arr = lens_softness(arr, lens_strength*0.18)
        arr = halation_pass(arr, lens_strength*0.08)

    elif lens_name == "24mm Architecture":
        arr = lens_softness(arr, lens_strength*0.08)
        arr = chromatic_aberration(arr, lens_strength*0.05)

    elif lens_name == "Cinema Neutral":
        arr = lens_softness(arr, lens_strength*0.15)
        arr = halation_pass(arr, lens_strength*0.06)
        arr = atmospheric_depth(arr, lens_strength*0.04)

    return arr


def _luminance(arr):
    x = arr.astype(np.float32) / 255.0
    return 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]

def protect_sky(arr, amount):
    if amount <= 0 or not CV2_AVAILABLE:
        return arr
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[...,0]
    s = hsv[...,1] / 255.0
    v = hsv[...,2] / 255.0
    sky = (((h > 85) & (h < 130) & (s > 0.10) & (v > 0.35))).astype(np.float32)
    sky = cv2.GaussianBlur(sky, (0,0), sigmaX=4.0)
    # keep sky from getting overly saturated/contrasty
    edited = arr.copy().astype(np.float32)
    edited[...,2] *= 1.0 + amount*0.035
    edited[...,0] *= 1.0 - amount*0.015
    return _blend_by_mask(arr, np.clip(edited,0,255).astype(np.uint8), sky, amount*0.50)

def deepen_shadows(arr, amount):
    if amount <= 0:
        return arr
    x = arr.astype(np.float32) / 255.0
    lum = _luminance(arr)
    mask = np.clip((0.55-lum)/0.55, 0, 1)[...,None]
    y = x * (1.0 - mask*amount*0.18)
    return np.clip(y*255,0,255).astype(np.uint8)

def green_balance(arr, amount):
    if amount == 0 or not CV2_AVAILABLE:
        return arr
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[...,0]
    s = hsv[...,1] / 255.0
    green = ((h > 28) & (h < 95) & (s > 0.16)).astype(np.float32)
    green = cv2.GaussianBlur(green, (0,0), sigmaX=3.0)
    edited = arr.astype(np.float32).copy()
    if amount > 0:
        # richer but still natural
        edited[...,1] *= 1.0 + amount*0.035
        edited[...,0] *= 1.0 - amount*0.010
    else:
        # mute neon CGI greens
        a = abs(amount)
        edited[...,1] *= 1.0 - a*0.060
        edited[...,0] *= 1.0 + a*0.015
    return _blend_by_mask(arr, np.clip(edited,0,255).astype(np.uint8), green, min(1.0, abs(amount))*0.70)

def anti_cgi_color(arr, amount):
    if amount <= 0:
        return arr
    img = np_to_pil(arr)
    # Slightly restrain global saturation and harsh contrast.
    img = ImageEnhance.Color(img).enhance(1.0 - amount*0.10)
    img = ImageEnhance.Contrast(img).enhance(1.0 - amount*0.035)
    out = np.asarray(img).astype(np.float32)
    # pull pure RGB extremes inward a little, like real sensor/color science
    mean = out.mean(axis=2, keepdims=True)
    out = mean + (out-mean)*(1.0 - amount*0.045)
    return np.clip(out,0,255).astype(np.uint8)

def grade_preset(arr, name, strength):
    if strength <= 0:
        return arr

    x = arr.astype(np.float32)
    if name == "Natural Daylight":
        x[...,0] *= 1.0 + 0.010*strength
        x[...,2] *= 1.0 - 0.006*strength

    elif name == "Golden Hour":
        x[...,0] *= 1.0 + 0.060*strength
        x[...,1] *= 1.0 + 0.018*strength
        x[...,2] *= 1.0 - 0.045*strength

    elif name == "Overcast":
        x[...,0] *= 1.0 - 0.010*strength
        x[...,1] *= 1.0 + 0.004*strength
        x[...,2] *= 1.0 + 0.018*strength

    elif name == "Blue Hour":
        x[...,0] *= 1.0 - 0.035*strength
        x[...,1] *= 1.0 + 0.005*strength
        x[...,2] *= 1.0 + 0.055*strength

    elif name == "Cinematic Neutral":
        # slightly cool shadows, restrained warm highlights
        lum = _luminance(arr)
        shadow = np.clip((0.55-lum)/0.55,0,1)[...,None]
        highlight = np.clip((lum-0.55)/0.45,0,1)[...,None]
        x[...,2] *= 1.0 + shadow[...,0]*0.030*strength
        x[...,0] *= 1.0 + highlight[...,0]*0.024*strength
        x[...,1] *= 1.0 - 0.008*strength

    return np.clip(x,0,255).astype(np.uint8)

def color_grade_pass(
    arr,
    grade_name,
    grade_strength,
    sky_protection,
    shadow_depth,
    green_tone,
    anti_cgi,
):
    base = grade_preset(arr, grade_name, grade_strength)
    base = protect_sky(base, sky_protection)
    base = deepen_shadows(base, shadow_depth)
    base = green_balance(base, green_tone)
    base = anti_cgi_color(base, anti_cgi)
    return base


def analyze_image_characteristics(image):
    arr = pil_to_np(image)
    x = arr.astype(np.float32) / 255.0
    lum = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]

    highlights = float((lum > 0.92).mean())
    deep_shadows = float((lum < 0.08).mean())
    mean_lum = float(lum.mean())
    contrast_std = float(lum.std())

    maxc = x.max(axis=2)
    minc = x.min(axis=2)
    sat = np.where(maxc > 1e-6, (maxc-minc)/(maxc+1e-6), 0)
    mean_sat = float(sat.mean())

    # Edge energy as a simple sharpness / crispness proxy.
    gray = (lum*255).astype(np.uint8)
    if CV2_AVAILABLE:
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        edge_energy = float(lap.var())
    else:
        edge_energy = float(np.var(np.diff(gray.astype(np.float32), axis=1)))

    # Green dominance.
    if CV2_AVAILABLE:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h = hsv[...,0]
        s = hsv[...,1] / 255.0
        green_mask = ((h > 28) & (h < 95) & (s > 0.18))
        green_ratio = float(green_mask.mean())
        if green_mask.any():
            green_sat = float(s[green_mask].mean())
        else:
            green_sat = 0.0
    else:
        green_ratio = 0.0
        green_sat = 0.0

    # Heuristic "CGI cleanliness": high sharpness + low noise + moderate/high saturation.
    # This is not a classifier; only a practical tuning hint.
    cleanliness = 0.0
    cleanliness += min(edge_energy / 800.0, 1.0) * 0.35
    cleanliness += max(0.0, (mean_sat - 0.18) / 0.30) * 0.30
    cleanliness += max(0.0, (contrast_std - 0.18) / 0.20) * 0.20
    cleanliness += (1.0 - min(highlights*5.0, 1.0)) * 0.15
    cleanliness = float(np.clip(cleanliness, 0, 1))

    recommendations = {
        "grade_name": "Natural Daylight",
        "grade_strength": 0.26,
        "sky_protection": 0.25,
        "shadow_depth": 0.16,
        "green_tone": -0.15,
        "anti_cgi": 0.28,
        "denoise": 0.08,
        "texture": 0.18,
        "sharp": 0.12,
        "grain": 0.04,
        "rolloff": 0.30,
    }

    notes = []

    if highlights > 0.08:
        recommendations["sky_protection"] = 0.38
        recommendations["rolloff"] = 0.45
        notes.append("Есть заметные пересветы — усиливаю защиту светов и highlight roll-off.")
    elif highlights < 0.01:
        recommendations["sky_protection"] = 0.18

    if deep_shadows > 0.12:
        recommendations["shadow_depth"] = 0.08
        notes.append("Много очень тёмных областей — не углубляю тени дополнительно.")
    elif contrast_std < 0.16:
        recommendations["shadow_depth"] = 0.24
        notes.append("Кадр выглядит плоским — рекомендую чуть глубже тени.")

    if green_ratio > 0.20 and green_sat > 0.45:
        recommendations["green_tone"] = -0.28
        notes.append("Зелени много и она насыщенная — приглушаю CGI-зелёный.")
    elif green_ratio < 0.05:
        recommendations["green_tone"] = -0.05

    if cleanliness > 0.62:
        recommendations["anti_cgi"] = 0.40
        recommendations["grain"] = 0.055
        recommendations["texture"] = 0.22
        recommendations["sharp"] = 0.08
        notes.append("Кадр выглядит слишком цифровым/стерильным — усиливаю anti-CGI и микрофактуру, снижаю финальную резкость.")
    elif cleanliness < 0.35:
        recommendations["anti_cgi"] = 0.18
        recommendations["grain"] = 0.025

    if edge_energy < 120:
        recommendations["sharp"] = 0.18
        notes.append("Исходник мягкий — можно слегка поднять финальную резкость.")
    elif edge_energy > 700:
        recommendations["sharp"] = 0.06
        notes.append("Исходник уже очень резкий — дополнительную резкость лучше держать низкой.")

    if mean_sat > 0.34:
        notes.append("Общая насыщенность высокая — anti-CGI color поможет сделать цвет натуральнее.")

    if not notes:
        notes.append("Кадр сбалансирован. Рекомендую мягкую обработку без сильных коррекций.")

    metrics = {
        "Средняя яркость": mean_lum,
        "Контраст": contrast_std,
        "Пересветы": highlights,
        "Глубокие тени": deep_shadows,
        "Средняя насыщенность": mean_sat,
        "Доля зелени": green_ratio,
        "Насыщенность зелени": green_sat,
        "Резкость/edge energy": edge_energy,
        "CGI-cleanliness": cleanliness,
    }
    return metrics, recommendations, notes


def scale_recommendations(recs, mode):
    """
    Scale auto-analysis recommendations into three practical strengths.
    Safe preserves architecture and avoids aggressive tonal changes.
    Photo is the balanced default.
    Maximum Realism pushes anti-CGI, texture and optical character further.
    """
    out = dict(recs)

    if mode == "Safe":
        factors = {
            "grade_strength": 0.72,
            "sky_protection": 0.85,
            "shadow_depth": 0.70,
            "green_tone": 0.70,
            "anti_cgi": 0.62,
            "denoise": 0.75,
            "texture": 0.65,
            "sharp": 0.78,
            "grain": 0.55,
            "rolloff": 0.78,
        }
    elif mode == "Maximum Realism":
        factors = {
            "grade_strength": 1.18,
            "sky_protection": 1.08,
            "shadow_depth": 1.12,
            "green_tone": 1.20,
            "anti_cgi": 1.30,
            "denoise": 1.05,
            "texture": 1.28,
            "sharp": 0.88,
            "grain": 1.35,
            "rolloff": 1.18,
        }
    else:
        factors = {
            "grade_strength": 1.00,
            "sky_protection": 1.00,
            "shadow_depth": 1.00,
            "green_tone": 1.00,
            "anti_cgi": 1.00,
            "denoise": 1.00,
            "texture": 1.00,
            "sharp": 1.00,
            "grain": 1.00,
            "rolloff": 1.00,
        }

    for k, f in factors.items():
        if k in out:
            out[k] = out[k] * f

    out["grade_strength"] = float(np.clip(out["grade_strength"], 0, 1))
    out["sky_protection"] = float(np.clip(out["sky_protection"], 0, 1))
    out["shadow_depth"] = float(np.clip(out["shadow_depth"], 0, 1))
    out["green_tone"] = float(np.clip(out["green_tone"], -1, 1))
    out["anti_cgi"] = float(np.clip(out["anti_cgi"], 0, 1))
    out["denoise"] = float(np.clip(out["denoise"], 0, 1))
    out["texture"] = float(np.clip(out["texture"], 0, 1))
    out["sharp"] = float(np.clip(out["sharp"], 0, 1))
    out["grain"] = float(np.clip(out["grain"], 0, 1))
    out["rolloff"] = float(np.clip(out["rolloff"], 0, 1))
    return out

def realism_score(image):
    """
    Heuristic photo-likeness score for comparing versions of the SAME image.
    It is not a trained classifier and should not be interpreted as an objective
    measure of whether an image is "real".
    """
    metrics, _, _ = analyze_image_characteristics(image)

    highlight_penalty = min(metrics["Пересветы"] / 0.12, 1.0)
    shadow_penalty = min(metrics["Глубокие тени"] / 0.18, 1.0)

    sat = metrics["Средняя насыщенность"]
    sat_penalty = min(abs(sat - 0.24) / 0.28, 1.0)

    contrast = metrics["Контраст"]
    contrast_penalty = min(abs(contrast - 0.20) / 0.20, 1.0)

    green_penalty = 0.0
    if metrics["Доля зелени"] > 0.16:
        green_penalty = min(
            max(0.0, metrics["Насыщенность зелени"] - 0.42) / 0.35,
            1.0
        )

    edge = metrics["Резкость/edge energy"]
    if edge < 70:
        sharp_penalty = min((70-edge)/70, 1.0)
    elif edge > 950:
        sharp_penalty = min((edge-950)/950, 1.0)
    else:
        sharp_penalty = 0.0

    cleanliness_penalty = metrics["CGI-cleanliness"]

    penalty = (
        highlight_penalty*0.16 +
        shadow_penalty*0.10 +
        sat_penalty*0.14 +
        contrast_penalty*0.12 +
        green_penalty*0.12 +
        sharp_penalty*0.10 +
        cleanliness_penalty*0.26
    )
    photo_score = int(round(np.clip(100*(1.0-penalty), 0, 100)))
    render_score = 100 - photo_score
    return render_score, photo_score, metrics


def make_split_preview(before, after, split_percent=50):
    """
    Create a single preview image:
    left = before, right = after, split controlled by percentage.
    """
    before = before.convert("RGB")
    after = after.convert("RGB")

    if after.size != before.size:
        after = after.resize(before.size, Image.Resampling.LANCZOS)

    w, h = before.size
    x = int(np.clip(split_percent / 100.0, 0, 1) * w)

    out = before.copy()
    if x < w:
        out.paste(after.crop((x, 0, w, h)), (x, 0))

    # center divider
    arr = np.asarray(out).copy()
    if 1 <= x < w-1:
        line = max(1, w // 900)
        arr[:, max(0,x-line):min(w,x+line), :] = 245
    return Image.fromarray(arr)


def score_feedback(before_score, after_score):
    delta = after_score - before_score
    if delta >= 12:
        return "Сильное улучшение по текущей эвристике."
    if delta >= 5:
        return "Заметное улучшение по текущей эвристике."
    if delta >= 1:
        return "Небольшое улучшение. Можно тонко донастроить материалы и цвет."
    if delta == 0:
        return "По метрикам изменения нейтральны."
    return "Текущая обработка ухудшила часть метрик. Попробуйте более мягкий режим."


def build_variant_settings(base_recs, variant_name):
    """
    Build four distinct non-generative treatment profiles from auto-analysis recommendations.
    """
    r = dict(base_recs)

    if variant_name == "Safe":
        r = scale_recommendations(r, "Safe")
        return {
            "strength": 0.26,
            "local_contrast": 0.16,
            "texture": r["texture"],
            "shadows": 0.06,
            "highlights": -0.08,
            "warmth": 0.00,
            "tint": 0.00,
            "saturation": -0.02,
            "denoise": r["denoise"],
            "sharp": r["sharp"],
            "grain": r["grain"],
            "bloom": 0.00,
            "glass": 0.18,
            "facade": 0.24,
            "metal": 0.12,
            "wood": 0.12,
            "greenery": 0.16,
            "skin": 0.08,
            "lens_name": "24mm Architecture",
            "lens_strength": 0.16,
            "rolloff": r["rolloff"],
            "halation": 0.01,
            "chroma_ab": 0.01,
            "depth_haze": 0.04,
            "lens_soft": 0.05,
            "sensor_noise_amount": 0.02,
            "grade_name": r["grade_name"],
            "grade_strength": r["grade_strength"],
            "sky_protection": r["sky_protection"],
            "shadow_depth": r["shadow_depth"],
            "green_tone": r["green_tone"],
            "anti_cgi": r["anti_cgi"],
        }

    if variant_name == "Natural Photo":
        r = scale_recommendations(r, "Photo")
        return {
            "strength": 0.40,
            "local_contrast": 0.24,
            "texture": r["texture"],
            "shadows": 0.10,
            "highlights": -0.12,
            "warmth": 0.02,
            "tint": 0.00,
            "saturation": -0.03,
            "denoise": r["denoise"],
            "sharp": r["sharp"],
            "grain": r["grain"],
            "bloom": 0.01,
            "glass": 0.26,
            "facade": 0.32,
            "metal": 0.16,
            "wood": 0.16,
            "greenery": 0.22,
            "skin": 0.10,
            "lens_name": "50mm Natural",
            "lens_strength": 0.24,
            "rolloff": r["rolloff"],
            "halation": 0.025,
            "chroma_ab": 0.018,
            "depth_haze": 0.06,
            "lens_soft": 0.08,
            "sensor_noise_amount": 0.035,
            "grade_name": "Natural Daylight",
            "grade_strength": r["grade_strength"],
            "sky_protection": r["sky_protection"],
            "shadow_depth": r["shadow_depth"],
            "green_tone": r["green_tone"],
            "anti_cgi": r["anti_cgi"],
        }

    if variant_name == "Cinematic":
        r = scale_recommendations(r, "Photo")
        return {
            "strength": 0.44,
            "local_contrast": 0.26,
            "texture": min(1.0, r["texture"]*1.08),
            "shadows": 0.06,
            "highlights": -0.16,
            "warmth": 0.01,
            "tint": -0.01,
            "saturation": -0.06,
            "denoise": r["denoise"],
            "sharp": max(0.04, r["sharp"]*0.80),
            "grain": min(1.0, r["grain"]*1.25),
            "bloom": 0.02,
            "glass": 0.24,
            "facade": 0.30,
            "metal": 0.14,
            "wood": 0.18,
            "greenery": 0.20,
            "skin": 0.10,
            "lens_name": "Cinema Neutral",
            "lens_strength": 0.32,
            "rolloff": min(1.0, r["rolloff"]*1.12),
            "halation": 0.045,
            "chroma_ab": 0.022,
            "depth_haze": 0.075,
            "lens_soft": 0.11,
            "sensor_noise_amount": 0.045,
            "grade_name": "Cinematic Neutral",
            "grade_strength": min(1.0, r["grade_strength"]*1.08),
            "sky_protection": r["sky_protection"],
            "shadow_depth": min(1.0, r["shadow_depth"]*1.08),
            "green_tone": r["green_tone"],
            "anti_cgi": min(1.0, r["anti_cgi"]*1.10),
        }

    # Maximum Realism
    r = scale_recommendations(r, "Maximum Realism")
    return {
        "strength": 0.54,
        "local_contrast": 0.32,
        "texture": r["texture"],
        "shadows": 0.12,
        "highlights": -0.18,
        "warmth": 0.015,
        "tint": 0.00,
        "saturation": -0.05,
        "denoise": r["denoise"],
        "sharp": r["sharp"],
        "grain": r["grain"],
        "bloom": 0.02,
        "glass": 0.34,
        "facade": 0.42,
        "metal": 0.22,
        "wood": 0.22,
        "greenery": 0.28,
        "skin": 0.12,
        "lens_name": "35mm Documentary",
        "lens_strength": 0.34,
        "rolloff": r["rolloff"],
        "halation": 0.05,
        "chroma_ab": 0.025,
        "depth_haze": 0.08,
        "lens_soft": 0.10,
        "sensor_noise_amount": 0.05,
        "grade_name": "Cinematic Neutral",
        "grade_strength": r["grade_strength"],
        "sky_protection": r["sky_protection"],
        "shadow_depth": r["shadow_depth"],
        "green_tone": r["green_tone"],
        "anti_cgi": r["anti_cgi"],
    }


def render_variant(image, settings):
    return process_free(
        image=image,
        strength=settings["strength"],
        local_contrast=settings["local_contrast"],
        texture=settings["texture"],
        shadows=settings["shadows"],
        highlights=settings["highlights"],
        warmth=settings["warmth"],
        tint=settings["tint"],
        saturation=settings["saturation"],
        denoise=settings["denoise"],
        sharp=settings["sharp"],
        grain=settings["grain"],
        bloom=settings["bloom"],
        glass=settings["glass"],
        facade=settings["facade"],
        metal=settings["metal"],
        wood=settings["wood"],
        greenery=settings["greenery"],
        skin=settings["skin"],
        lens_name=settings["lens_name"],
        lens_strength=settings["lens_strength"],
        rolloff=settings["rolloff"],
        halation=settings["halation"],
        chroma_ab=settings["chroma_ab"],
        depth_haze=settings["depth_haze"],
        lens_soft=settings["lens_soft"],
        sensor_noise_amount=settings["sensor_noise_amount"],
        grade_name=settings["grade_name"],
        grade_strength=settings["grade_strength"],
        sky_protection=settings["sky_protection"],
        shadow_depth=settings["shadow_depth"],
        green_tone=settings["green_tone"],
        anti_cgi=settings["anti_cgi"],
        preserve_geometry=True,
    )

def run_full_pipeline(
    image,
    realism_mode="Photo",
    use_ai_if_available=True,
    face_restore_mode="Auto",
    final_4k=True,
):
    """
    End-to-end non-generative + optional open-source AI pipeline.
    It never requires AI to succeed: unavailable modules are skipped.
    """
    logs = []

    metrics, recs, notes = analyze_image_characteristics(image)
    logs.append("1. Автоанализ выполнен.")

    if realism_mode == "Safe":
        variant_name = "Safe"
    elif realism_mode == "Maximum Realism":
        variant_name = "Maximum Realism"
    else:
        variant_name = "Natural Photo"

    settings = build_variant_settings(recs, variant_name)
    result = render_variant(image, settings)
    logs.append(f"2. Photoreal-pass: {variant_name}.")

    status = ai_status()

    if use_ai_if_available and status.get("realesrgan"):
        ai_result, msg = run_ai_restore(
            result,
            module="Real-ESRGAN · Detail + Upscale",
            scale=2,
            tile=400,
            fidelity=0.80,
            face_only=False,
        )
        if ai_result is not None:
            result = ai_result
            logs.append("3. Real-ESRGAN 2x выполнен.")
        else:
            logs.append("3. Real-ESRGAN пропущен: " + msg)
    else:
        logs.append("3. Real-ESRGAN недоступен или отключён — пропущен.")

    # Face restoration only when explicitly requested or Auto + a backend is available.
    do_face = False
    chosen_face_module = None

    if face_restore_mode in ("None", "Не использовать"):
        do_face = False
        chosen_face_module = None
    elif face_restore_mode == "CodeFormer" and status.get("codeformer"):
        do_face = True
        chosen_face_module = "CodeFormer · Faces / Identity"
    elif face_restore_mode == "GFPGAN" and status.get("gfpgan"):
        do_face = True
        chosen_face_module = "GFPGAN · Faces + Background"
    elif face_restore_mode == "Auto":
        # Conservative Auto: only if a face backend exists.
        if status.get("codeformer"):
            do_face = True
            chosen_face_module = "CodeFormer · Faces / Identity"
        elif status.get("gfpgan"):
            do_face = True
            chosen_face_module = "GFPGAN · Faces + Background"

    if do_face and chosen_face_module:
        face_result, msg = run_ai_restore(
            result,
            module=chosen_face_module,
            scale=1,
            tile=300,
            fidelity=0.88,
            face_only=True,
        )
        if face_result is not None:
            result = face_result
            logs.append(f"4. Face restore: {chosen_face_module.split(' · ')[0]}.")
        else:
            logs.append("4. Face restore пропущен: " + msg)
    else:
        logs.append("4. Face restore не применялся.")

    # Final restrained photographic finish after AI.
    arr = pil_to_np(result)
    arr = color_grade_pass(
        arr,
        grade_name="Cinematic Neutral" if realism_mode == "Maximum Realism" else "Natural Daylight",
        grade_strength=0.18 if realism_mode == "Safe" else 0.24,
        sky_protection=0.22,
        shadow_depth=0.10 if realism_mode == "Safe" else 0.16,
        green_tone=-0.10,
        anti_cgi=0.16 if realism_mode == "Safe" else 0.24,
    )
    arr = apply_highlight_rolloff(arr, 0.22)
    arr = add_grain(arr, 0.025 if realism_mode == "Safe" else 0.04)
    result = np_to_pil(arr)
    logs.append("5. Финальный Color Grade выполнен.")

    if final_4k:
        result = resize_long_edge(result, "4K")
        logs.append("6. Экспорт подготовлен в 4K.")
    else:
        logs.append("6. Размер оставлен без принудительного 4K.")

    _, final_score, _ = realism_score(result)
    logs.append(f"Photo Score результата: {final_score}/100.")

    return result, logs, final_score

def auto_object_masks(image):
    """
    Lightweight CPU masks for common architectural regions.
    These are conservative heuristics, not semantic-AI segmentation.
    """
    arr = pil_to_np(image)
    h, w = arr.shape[:2]
    masks = {}

    if CV2_AVAILABLE:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hh = hsv[...,0]
        ss = hsv[...,1] / 255.0
        vv = hsv[...,2] / 255.0

        sky = ((hh > 85) & (hh < 135) & (ss > 0.08) & (vv > 0.35)).astype(np.float32)
        green = ((hh > 28) & (hh < 95) & (ss > 0.18)).astype(np.float32)
        road = ((ss < 0.22) & (vv > 0.15) & (vv < 0.78)).astype(np.float32)
        glass = (((ss < 0.34) & (vv > 0.28)) | ((hh > 85) & (hh < 130) & (vv > 0.25))).astype(np.float32)
        facade = ((vv > 0.16) & (vv < 0.93) & (~(green > 0))).astype(np.float32)

        # position priors
        yy = np.linspace(0, 1, h, dtype=np.float32)[:,None]
        sky *= (yy < 0.58)
        road *= (yy > 0.45)

        for name, m in [("sky",sky),("greenery",green),("road",road),("glass",glass),("facade",facade)]:
            masks[name] = cv2.GaussianBlur(m, (0,0), sigmaX=3.0)
    else:
        z = np.zeros((h,w), dtype=np.float32)
        for name in ["sky","greenery","road","glass","facade"]:
            masks[name] = z.copy()

    return masks

def apply_masked_scene_pass(
    image,
    sky_strength=0.20,
    greenery_strength=0.18,
    road_strength=0.12,
    glass_strength=0.22,
    facade_strength=0.24,
):
    arr = pil_to_np(image)
    masks = auto_object_masks(image)
    base = arr.copy()

    # sky: restrained saturation + highlight control
    sky_edit = protect_highlights(base, 0.35)
    sky_edit = adjust_temperature(sky_edit, -0.03, 0.0)
    base = _blend_by_mask(base, sky_edit, masks["sky"], sky_strength)

    # greenery: suppress neon green
    green_edit = green_balance(base, -0.25)
    base = _blend_by_mask(base, green_edit, masks["greenery"], greenery_strength)

    # road: slightly darker and more local texture
    road_edit = deepen_shadows(base, 0.12)
    road_edit = add_texture(road_edit, 0.18)
    base = _blend_by_mask(base, road_edit, masks["road"], road_strength)

    # glass: clarity/reflection separation
    glass_edit = cv_local_contrast(base, 0.28)
    glass_edit = adjust_temperature(glass_edit, -0.02, 0.0)
    base = _blend_by_mask(base, glass_edit, masks["glass"], glass_strength)

    # facade: local texture and controlled contrast
    fac_edit = cv_local_contrast(base, 0.24)
    fac_edit = add_texture(fac_edit, 0.22)
    base = _blend_by_mask(base, fac_edit, masks["facade"], facade_strength)

    return np_to_pil(base)
def final_sharpen(image, amount):
    if amount <= 0:
        return image
    return image.filter(ImageFilter.UnsharpMask(
        radius=1.0,
        percent=int(35 + amount*115),
        threshold=4
    ))

def process_free(
    image,
    strength,
    local_contrast,
    texture,
    shadows,
    highlights,
    warmth,
    tint,
    saturation,
    denoise,
    sharp,
    grain,
    bloom,
    glass,
    facade,
    metal,
    wood,
    greenery,
    skin,
    lens_name,
    lens_strength,
    rolloff,
    halation,
    chroma_ab,
    depth_haze,
    lens_soft,
    sensor_noise_amount,
    grade_name,
    grade_strength,
    sky_protection,
    shadow_depth,
    green_tone,
    anti_cgi,
    preserve_geometry,
):
    arr = pil_to_np(image)

    # Geometry-preserving CPU pass: only tonal/color/detail operations.
    arr = cv_denoise(arr, denoise * (0.75 if preserve_geometry else 1.0))
    arr = lift_shadows(arr, shadows)
    arr = protect_highlights(arr, max(0.0, -highlights))
    arr = cv_local_contrast(arr, local_contrast * strength)
    arr = adjust_temperature(arr, warmth, tint)
    arr = add_texture(arr, texture * strength)
    arr = material_pass(arr, glass, facade, metal, wood, greenery, skin)
    arr = apply_highlight_rolloff(arr, rolloff)
    arr = lens_softness(arr, lens_soft)
    arr = apply_lens_profile(arr, lens_name, lens_strength)
    arr = halation_pass(arr, halation)
    arr = chromatic_aberration(arr, chroma_ab)
    arr = atmospheric_depth(arr, depth_haze)
    arr = sensor_noise(arr, sensor_noise_amount)
    arr = color_grade_pass(
        arr,
        grade_name=grade_name,
        grade_strength=grade_strength,
        sky_protection=sky_protection,
        shadow_depth=shadow_depth,
        green_tone=green_tone,
        anti_cgi=anti_cgi,
    )
    arr = add_bloom(arr, bloom)
    arr = add_grain(arr, grain)

    result = np_to_pil(arr)
    result = ImageEnhance.Color(result).enhance(max(0.65, 1.0 + saturation*0.45))
    result = ImageEnhance.Contrast(result).enhance(1.0 + strength*0.035)
    result = final_sharpen(result, sharp)
    return result

uploaded = st.file_uploader("Загрузите CGI / AI / 3D-кадр", type=["png","jpg","jpeg","webp"])

with st.sidebar:
    st.header("Профиль")
    preset_name = st.selectbox("Тип кадра", list(PRESETS))
    p = PRESETS[preset_name]

    st.header("Защита сцены")
    protect_architecture = st.checkbox("Максимально сохранять архитектуру", True)
    protect_people = st.checkbox("Не менять людей", True)
    preserve_text = st.checkbox("Не менять текст / логотипы", True)
    st.caption("В бесплатном CPU-режиме геометрия и объекты не генерируются заново, поэтому защита максимальная по умолчанию.")

    st.header("Экспорт")
    export_size = st.selectbox("Размер", ["Оригинал", "2K", "4K"], index=1)
    export_format = st.selectbox("Формат", ["PNG", "JPEG"], index=0)
    jpeg_quality = st.slider("JPEG качество", 85, 100, 96) if export_format == "JPEG" else 96

tab_full, tab0, tab1, tab2, tab3, tab_masks, tab4, tab5, tab6, tab7, tab_ai, tab_ext, tab8, tab9, tab10 = st.tabs([
    "🚀 FULL PIPELINE",
    "🩺 Автоанализ",
    "⚡ Фотореализм",
    "🔎 Качество",
    "🧱 Материалы",
    "🎯 Маски объектов",
    "📷 Камера",
    "🌤 Свет / Color Grade",
    "🎞 Оптика / плёнка",
    "🧠 Бесплатные AI-модули",
    "🤖 AI Restore",
    "🔌 Pro Providers",
    "🧪 4 варианта",
    "↔️ Сравнение",
    "ℹ️ Что делает режим",
])


with tab_full:
    st.subheader("FULL REALPHOTO PIPELINE")
    st.caption("Одна кнопка: анализ → photoreal → AI detail restore при наличии → лица → финальный цвет → 4K.")

    c1, c2 = st.columns(2)
    with c1:
        full_mode = st.radio(
            "Сила обработки",
            ["Safe", "Photo", "Maximum Realism"],
            horizontal=True,
            index=1,
        )
        full_use_ai = st.checkbox("Использовать AI-модули, если доступны", True)
    with c2:
        face_mode = st.selectbox(
            "Face restore",
            ["Auto", "Не использовать", "CodeFormer", "GFPGAN"],
            index=0,
        )
        full_4k = st.checkbox("Финальный экспорт 4K", True)

    if uploaded is None:
        st.info("Загрузите изображение.")
    else:
        source_full = Image.open(uploaded).convert("RGB")
        st.image(source_full, caption="Исходник", use_container_width=True)

        if st.button("FULL REALPHOTO PIPELINE", type="primary", use_container_width=True):
            frm = "None" if face_mode == "Не использовать" else face_mode
            result_full, logs_full, score_full = run_full_pipeline(
                source_full,
                realism_mode=full_mode,
                use_ai_if_available=full_use_ai,
                face_restore_mode=frm,
                final_4k=full_4k,
            )
            st.session_state["full_pipeline_result"] = result_full
            st.session_state["full_pipeline_logs"] = logs_full
            st.session_state["full_pipeline_score"] = score_full
            st.session_state["realphoto_result"] = result_full

        if "full_pipeline_result" in st.session_state:
            result_full = st.session_state["full_pipeline_result"]
            st.image(
                result_full,
                caption=f"FULL PIPELINE · Photo Score {st.session_state.get('full_pipeline_score', 0)}/100",
                use_container_width=True,
            )

            st.markdown("#### Что было сделано")
            for line in st.session_state.get("full_pipeline_logs", []):
                st.write("• " + line)

            fb = io.BytesIO()
            result_full.save(fb, format="PNG")
            st.download_button(
                "СКАЧАТЬ FULL RESULT",
                fb.getvalue(),
                "realphoto_full_pipeline_4k.png",
                "image/png",
                use_container_width=True,
            )


with tab0:
    st.subheader("Автоанализ исходного кадра")
    st.caption("Это безопасный диагностический режим. Он ничего не меняет сам — только измеряет изображение и предлагает настройки.")

    if uploaded is None:
        st.info("Сначала загрузите изображение.")
    else:
        if st.button("ПРОАНАЛИЗИРОВАТЬ КАДР", use_container_width=True):
            metrics, recs, notes = analyze_image_characteristics(Image.open(uploaded).convert("RGB"))
            st.session_state["analysis_metrics"] = metrics
            st.session_state["analysis_recs"] = recs
            st.session_state["analysis_notes"] = notes

        if "analysis_metrics" in st.session_state:
            metrics = st.session_state["analysis_metrics"]
            recs = st.session_state["analysis_recs"]
            notes = st.session_state["analysis_notes"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Пересветы", f"{metrics['Пересветы']*100:.1f}%")
            c2.metric("Глубокие тени", f"{metrics['Глубокие тени']*100:.1f}%")
            c3.metric("CGI-cleanliness", f"{metrics['CGI-cleanliness']*100:.0f}%")

            c4, c5, c6 = st.columns(3)
            c4.metric("Насыщенность", f"{metrics['Средняя насыщенность']:.2f}")
            c5.metric("Доля зелени", f"{metrics['Доля зелени']*100:.1f}%")
            c6.metric("Резкость", f"{metrics['Резкость/edge energy']:.0f}")

            st.markdown("#### Что обнаружено")
            for n in notes:
                st.write("• " + n)

            st.markdown("#### Рекомендуемые настройки")
            st.code(
                "\\n".join([
                    f"Color Grade: {recs['grade_name']}",
                    f"Сила профиля: {recs['grade_strength']:.2f}",
                    f"Защита неба: {recs['sky_protection']:.2f}",
                    f"Глубина теней: {recs['shadow_depth']:.2f}",
                    f"Баланс зелени: {recs['green_tone']:.2f}",
                    f"Anti-CGI color: {recs['anti_cgi']:.2f}",
                    f"Denoise: {recs['denoise']:.2f}",
                    f"Фактура: {recs['texture']:.2f}",
                    f"Финальная резкость: {recs['sharp']:.2f}",
                    f"Зерно: {recs['grain']:.3f}",
                    f"Highlight roll-off: {recs['rolloff']:.2f}",
                ])
            )

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        strength = st.slider("Общая сила", 0.0, 1.0, float(st.session_state.get("strength", p["strength"])), 0.01, key="strength")
        local_contrast = st.slider("Локальный контраст", 0.0, 1.0, float(st.session_state.get("local_contrast", p["local_contrast"])), 0.01, key="local_contrast")
        shadows = st.slider("Тени", -1.0, 1.0, float(st.session_state.get("shadows", p["shadows"])), 0.01, key="shadows")
        highlights = st.slider("Света / пересветы", -1.0, 0.0, float(st.session_state.get("highlights", p["highlights"])), 0.01, key="highlights")
    with c2:
        warmth = st.slider("Температура", -1.0, 1.0, float(st.session_state.get("warmth", p["warmth"])), 0.01, key="warmth")
        tint = st.slider("Tint: зелёный ↔ пурпурный", -1.0, 1.0, float(st.session_state.get("tint", 0.0)), 0.01, key="tint")
        saturation = st.slider("Насыщенность", -1.0, 1.0, float(st.session_state.get("saturation", p["saturation"])), 0.01, key="saturation")
        denoise = st.slider("Убрать цифровой шум / мусор", 0.0, 1.0, float(st.session_state.get("denoise", p["denoise"])), 0.01, key="denoise")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        texture = st.slider("Фактура материалов / anti-plastic", 0.0, 1.0, float(st.session_state.get("texture", p["texture"])), 0.01, key="texture")
        sharp = st.slider("Финальная резкость", 0.0, 1.0, float(st.session_state.get("sharp", p["sharp"])), 0.01, key="sharp")
    with c2:
        st.metric("OpenCV", "Подключён" if CV2_AVAILABLE else "Недоступен")
        st.caption("OpenCV используется для локального контраста, шумоподавления и мягкого восстановления светотени.")

with tab3:
    st.caption("Локальные маски определяются автоматически по цвету и яркости. Для архитектурных кадров лучше начинать с 0.15–0.35.")
    c1, c2 = st.columns(2)
    with c1:
        glass = st.slider("Стекло / окна", 0.0, 1.0, 0.28, 0.01)
        facade = st.slider("Фасад / штукатурка / бетон", 0.0, 1.0, 0.34, 0.01)
        metal = st.slider("Металл / ограждения", 0.0, 1.0, 0.18, 0.01)
    with c2:
        wood = st.slider("Дерево / тёплые материалы", 0.0, 1.0, 0.16, 0.01)
        greenery = st.slider("Зелень / газон / деревья", 0.0, 1.0, 0.24, 0.01)
        skin = st.slider("Кожа человека", 0.0, 1.0, 0.10, 0.01)


with tab_masks:
    st.subheader("Автоматические маски объектов")
    st.caption("Лёгкие CPU-маски для архитектурных кадров. Они не перерисовывают сцену и подходят для бесплатного Streamlit.")

    if uploaded is None:
        st.info("Загрузите изображение.")
    else:
        mask_source = Image.open(uploaded).convert("RGB")
        masks = auto_object_masks(mask_source)

        c1, c2 = st.columns(2)
        with c1:
            mask_sky = st.slider("Небо", 0.0, 1.0, 0.20, 0.01)
            mask_green = st.slider("Зелень", 0.0, 1.0, 0.18, 0.01)
            mask_road = st.slider("Дорога", 0.0, 1.0, 0.12, 0.01)
        with c2:
            mask_glass = st.slider("Стекло / окна", 0.0, 1.0, 0.22, 0.01)
            mask_facade = st.slider("Фасад", 0.0, 1.0, 0.24, 0.01)

        if st.checkbox("Показать найденные маски"):
            names = ["sky","greenery","road","glass","facade"]
            cols = st.columns(5)
            for i, name in enumerate(names):
                m = (np.clip(masks[name],0,1)*255).astype(np.uint8)
                cols[i].image(m, caption=name, use_container_width=True)

        if st.button("ПРИМЕНИТЬ ОБРАБОТКУ ПО МАСКАМ", use_container_width=True):
            masked_result = apply_masked_scene_pass(
                mask_source,
                sky_strength=mask_sky,
                greenery_strength=mask_green,
                road_strength=mask_road,
                glass_strength=mask_glass,
                facade_strength=mask_facade,
            )
            st.session_state["masked_result"] = masked_result
            st.session_state["realphoto_result"] = masked_result

        if "masked_result" in st.session_state:
            st.image(st.session_state["masked_result"], caption="Результат по маскам", use_container_width=True)


with tab4:
    st.caption("Эти настройки имитируют характер реальной оптики и сенсора, не меняя геометрию сцены.")
    lens_name = st.selectbox("Профиль объектива", [
        "24mm Architecture",
        "35mm Documentary",
        "50mm Natural",
        "85mm Premium",
        "Cinema Neutral",
    ], index=0)
    lens_strength = st.slider("Сила профиля объектива", 0.0, 1.0, 0.28, 0.01)

    c1, c2 = st.columns(2)
    with c1:
        rolloff = st.slider("Highlight roll-off", 0.0, 1.0, float(st.session_state.get("rolloff", 0.34)), 0.01, key="rolloff")
        lens_soft = st.slider("Мягкость линзы", 0.0, 1.0, 0.10, 0.01)
        depth_haze = st.slider("Атмосферная глубина", 0.0, 1.0, 0.08, 0.01)
    with c2:
        halation = st.slider("Halation", 0.0, 0.5, 0.04, 0.01)
        chroma_ab = st.slider("Хроматическая аберрация", 0.0, 0.5, 0.03, 0.01)
        sensor_noise_amount = st.slider("Сенсорный шум", 0.0, 0.5, 0.04, 0.01)

    st.markdown("""
**Рекомендуемые стартовые профили**
- ЖК / экстерьер: `24mm Architecture`
- lifestyle / двор: `35mm Documentary`
- универсально: `50mm Natural`
- портрет / реклама: `85mm Premium`
- более киношно: `Cinema Neutral`
""")

with tab5:
    st.caption("Фотореалистичный цвет без перерисовки объектов.")
    grade_name = st.selectbox("Профиль света / цвета", [
        "Natural Daylight",
        "Golden Hour",
        "Overcast",
        "Blue Hour",
        "Cinematic Neutral",
    ], index=0)
    grade_strength = st.slider("Сила профиля", 0.0, 1.0, float(st.session_state.get("grade_strength", 0.28)), 0.01, key="grade_strength")

    c1, c2 = st.columns(2)
    with c1:
        sky_protection = st.slider("Защита неба", 0.0, 1.0, float(st.session_state.get("sky_protection", 0.28)), 0.01, key="sky_protection")
        shadow_depth = st.slider("Глубина теней", 0.0, 1.0, float(st.session_state.get("shadow_depth", 0.18)), 0.01, key="shadow_depth")
    with c2:
        green_tone = st.slider("Баланс зелени", -1.0, 1.0, float(st.session_state.get("green_tone", -0.18)), 0.01, key="green_tone")
        anti_cgi = st.slider("Anti-CGI color", 0.0, 1.0, float(st.session_state.get("anti_cgi", 0.30)), 0.01, key="anti_cgi")

    st.markdown("""
**Для архитектуры / ЖК**
- Natural Daylight: самый безопасный
- Golden Hour: сила 0.15–0.35
- Blue Hour: сила 0.15–0.30
- Cinematic Neutral: сила 0.20–0.35
- Anti-CGI color: обычно 0.20–0.40
- Баланс зелени: чаще от -0.10 до -0.30, чтобы убрать кислотный CGI-зелёный
""")

with tab6:
    grain = st.slider("Натуральное зерно", 0.0, 1.0, float(st.session_state.get("grain", p["grain"])), 0.01, key="grain")
    bloom = st.slider("Мягкий bloom на ярких участках", 0.0, 0.35, float(p["bloom"]), 0.01)
    st.caption("Для архитектуры bloom лучше держать 0–0.04, зерно 0.02–0.08.")

with tab7:
    st.subheader("Open-source модули для следующего этапа")
    st.markdown("""
**Real-ESRGAN** — AI upscale и восстановление деталей.  
**GFPGAN / CodeFormer** — восстановление лиц с контролем fidelity.  
**SwinIR** — super-resolution, denoise и JPEG artifact reduction.

На бесплатном Streamlit Community Cloud тяжёлые модели лучше не запускать постоянно:
им нужен большой RAM/VRAM и длительная загрузка весов. Интерфейс уже можно оставить здесь,
а GPU-модули запускать на отдельном backend или локальном компьютере.
""")
    use_realesrgan = st.checkbox("Real-ESRGAN (GPU backend)", disabled=True)
    use_face_ai = st.checkbox("GFPGAN / CodeFormer (GPU backend)", disabled=True)
    use_swinir = st.checkbox("SwinIR Restore (GPU backend)", disabled=True)



with tab_ai:
    st.subheader("AI Restore · Local / GPU")
    st.caption("Настоящее восстановление через open-source модели. Если модули не установлены, эта вкладка просто показывает статус и не ломает сайт.")

    status = ai_status()
    cols = st.columns(4)
    cols[0].metric("CUDA", "Да" if status["cuda"] else "Нет")
    cols[1].metric("Real-ESRGAN", "Готов" if status["realesrgan"] else "Нет")
    cols[2].metric("GFPGAN", "Готов" if status["gfpgan"] else "Нет")
    cols[3].metric("CodeFormer", "Готов" if status["codeformer"] else "Нет")

    st.write("SwinIR:", "Готов" if status["swinir"] else "не установлен")
    if status.get("device_name"):
        st.caption(f"Устройство: {status['device_name']}")

    ai_mode = st.selectbox(
        "AI-модуль",
        [
            "Real-ESRGAN · Detail + Upscale",
            "GFPGAN · Faces + Background",
            "CodeFormer · Faces / Identity",
            "SwinIR · Restore / Denoise",
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        ai_scale = st.select_slider("AI upscale", options=[1, 2, 4], value=2)
        tile_size = st.select_slider("Tile", options=[0, 200, 300, 400, 512], value=400)
    with c2:
        fidelity = st.slider(
            "CodeFormer fidelity",
            0.0, 1.0, 0.80, 0.05,
            help="Ближе к 1.0 = сильнее сохраняется исходная личность; ниже = модель исправляет лицо активнее."
        )
        face_only = st.checkbox("Только лица, не усиливать фон", False)

    st.warning(
        "Для архитектуры сначала используйте Real-ESRGAN или SwinIR. "
        "GFPGAN/CodeFormer имеет смысл включать только если в кадре действительно есть заметные лица."
    )

    if uploaded is None:
        st.info("Загрузите изображение.")
    else:
        if st.button("ЗАПУСТИТЬ AI RESTORE", type="primary", use_container_width=True):
            source_img = Image.open(uploaded).convert("RGB")
            result_ai, message = run_ai_restore(
                source_img,
                module=ai_mode,
                scale=ai_scale,
                tile=tile_size,
                fidelity=fidelity,
                face_only=face_only,
            )
            if result_ai is not None:
                st.session_state["ai_result"] = result_ai
                st.session_state["ai_message"] = message
            else:
                st.session_state["ai_message"] = message

        if "ai_message" in st.session_state:
            st.info(st.session_state["ai_message"])

        if "ai_result" in st.session_state:
            result_ai = st.session_state["ai_result"]
            st.image(result_ai, caption="AI Restore result", use_container_width=True)

            rb = io.BytesIO()
            result_ai.save(rb, format="PNG")
            st.download_button(
                "СКАЧАТЬ AI RESULT",
                rb.getvalue(),
                "realphoto_ai_restore.png",
                "image/png",
                use_container_width=True,
            )

            if st.button("ИСПОЛЬЗОВАТЬ AI RESULT КАК ОСНОВНОЙ", use_container_width=True):
                st.session_state["realphoto_result"] = result_ai
                st.success("AI-result установлен как основной результат.")



with tab_ext:
    st.subheader("Pro Providers · точки подключения для техотдела")
    st.caption("Эти блоки оставлены в новом сайте специально для будущего подключения API.")

    provider_status = external_provider_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("GPT Image", "Ключ найден" if provider_status["gpt_image"] else "Не настроен")
    c2.metric("Nano Banana", "Ключ найден" if provider_status["nano_banana"] else "Не настроен")
    c3.metric("Magnific", "Ключ найден" if provider_status["magnific"] else "Не настроен")

    provider = st.selectbox("Провайдер", ["GPT Image", "Nano Banana", "Magnific"])
    provider_prompt = st.text_area(
        "Инструкция для внешнего провайдера",
        value="Preserve architecture, geometry, people, text and logos. Improve photographic realism only.",
        height=120,
    )

    st.info(
        "Сейчас это безопасная интеграционная точка. "
        "Техотдел сможет реализовать API-вызовы в services/external_providers.py "
        "и хранить ключи в Streamlit Secrets / Secret Manager."
    )

    if uploaded is not None and st.button("ТЕСТ ПОДКЛЮЧЕНИЯ PROVIDER", use_container_width=True):
        source_ext = Image.open(uploaded).convert("RGB")
        out_ext, msg = run_external_provider(source_ext, provider, provider_prompt)
        st.info(msg)
        if out_ext is not None:
            st.image(out_ext, caption=f"{provider} result", use_container_width=True)


with tab8:
    st.subheader("4 автоматических варианта")
    st.caption("Один исходник → четыре негенеративные обработки. Архитектура и композиция не перерисовываются.")

    if uploaded is None:
        st.info("Сначала загрузите изображение.")
    else:
        if st.button("СОЗДАТЬ 4 ВАРИАНТА", type="primary", use_container_width=True):
            source_img = Image.open(uploaded).convert("RGB")
            _, base_recs, _ = analyze_image_characteristics(source_img)

            names = ["Safe", "Natural Photo", "Cinematic", "Maximum Realism"]
            generated = {}
            scores = {}
            for name in names:
                settings = build_variant_settings(base_recs, name)
                img_v = render_variant(source_img, settings)
                generated[name] = img_v
                _, score, _ = realism_score(img_v)
                scores[name] = score

            st.session_state["variant_images"] = generated
            st.session_state["variant_scores"] = scores

        if "variant_images" in st.session_state:
            generated = st.session_state["variant_images"]
            scores = st.session_state["variant_scores"]

            cols = st.columns(2)
            names = ["Safe", "Natural Photo", "Cinematic", "Maximum Realism"]

            for idx, name in enumerate(names):
                with cols[idx % 2]:
                    st.image(generated[name], caption=f"{name} · Photo Score {scores[name]}/100", use_container_width=True)

                    b = io.BytesIO()
                    generated[name].save(b, format="PNG")
                    st.download_button(
                        label=f"Скачать {name}",
                        data=b.getvalue(),
                        file_name=f"realphoto_{name.lower().replace(' ','_')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_variant_{name}"
                    )

            best_name = max(scores, key=scores.get)
            st.success(f"Лучший вариант по текущей эвристике: {best_name} · {scores[best_name]}/100")

            selected = st.radio(
                "Выбрать вариант как основной результат",
                names,
                horizontal=True,
                index=names.index(best_name),
            )
            if st.button("ИСПОЛЬЗОВАТЬ ВЫБРАННЫЙ ВАРИАНТ", use_container_width=True):
                st.session_state["realphoto_result"] = generated[selected]
                st.success(f"{selected} установлен как основной результат.")


with tab9:
    st.subheader("Before / After + Photo Score")
    st.caption("Score нужен только для сравнения вариантов одного и того же кадра. Это не AI-детектор и не объективная оценка реальности.")

    if uploaded is None:
        st.info("Загрузите изображение.")
    elif "realphoto_result" not in st.session_state:
        st.info("Сначала обработайте изображение кнопкой «СДЕЛАТЬ ФОТОГРАФИЧНЕЕ».")
    else:
        before_img = Image.open(uploaded).convert("RGB")
        after_img = st.session_state["realphoto_result"]

        b_render, b_photo, _ = realism_score(before_img)
        a_render, a_photo, _ = realism_score(after_img)

        c1, c2, c3 = st.columns(3)
        c1.metric("Photo Score · До", f"{b_photo}/100")
        c2.metric("Photo Score · После", f"{a_photo}/100", delta=f"{a_photo-b_photo:+d}")
        c3.metric("Render Score · После", f"{a_render}/100")

        st.write(score_feedback(b_photo, a_photo))

        split = st.slider("Before ↔ After", 0, 100, 50, 1, help="Слева исходник, справа результат.")
        preview = make_split_preview(before_img, after_img, split)
        st.image(preview, caption=f"Слева — исходник · Справа — результат · Разделитель {split}%", use_container_width=True)

        if st.checkbox("Показать технические метрики"):
            _, _, bm = realism_score(before_img)
            _, _, am = realism_score(after_img)
            rows = [
                ("Пересветы", bm["Пересветы"]*100, am["Пересветы"]*100, "%"),
                ("Глубокие тени", bm["Глубокие тени"]*100, am["Глубокие тени"]*100, "%"),
                ("Насыщенность", bm["Средняя насыщенность"], am["Средняя насыщенность"], ""),
                ("Контраст", bm["Контраст"], am["Контраст"], ""),
                ("Доля зелени", bm["Доля зелени"]*100, am["Доля зелени"]*100, "%"),
                ("CGI-cleanliness", bm["CGI-cleanliness"]*100, am["CGI-cleanliness"]*100, "%"),
                ("Edge energy", bm["Резкость/edge energy"], am["Резкость/edge energy"], ""),
            ]
            for name, bv, av, suffix in rows:
                st.write(f"**{name}:** {bv:.2f}{suffix} → {av:.2f}{suffix}")

with tab10:
    st.markdown("""
Этот режим **не перерисовывает** дома и людей. Он делает то, что ближе к работе фотографа/колориста:
- смягчает цифровую «стерильность» рендера;
- возвращает локальный контраст и материальность;
- аккуратнее удерживает света;
- раскрывает тени;
- добавляет микрофактуру и очень слабое зерно;
- делает финальную резкость после остальных операций.

Поэтому он подходит для кадров, где нельзя менять архитектуру, количество этажей,
дороги, людей, вывески и композицию.
""")

if uploaded is None:
    st.info("Загрузи кадр — появится сравнение «До / После».")
else:
    image = Image.open(uploaded).convert("RGB")
    st.markdown("### До / После")
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Исходник", use_container_width=True)

    if st.button("СДЕЛАТЬ ФОТОГРАФИЧНЕЕ", type="primary", use_container_width=True):
        result = process_free(
            image=image,
            strength=strength,
            local_contrast=local_contrast,
            texture=texture,
            shadows=shadows,
            highlights=highlights,
            warmth=warmth,
            tint=tint,
            saturation=saturation,
            denoise=denoise,
            sharp=sharp,
            grain=grain,
            bloom=bloom,
            glass=glass,
            facade=facade,
            metal=metal,
            wood=wood,
            greenery=greenery,
            skin=skin,
            lens_name=lens_name,
            lens_strength=lens_strength,
            rolloff=rolloff,
            halation=halation,
            chroma_ab=chroma_ab,
            depth_haze=depth_haze,
            lens_soft=lens_soft,
            sensor_noise_amount=sensor_noise_amount,
            grade_name=grade_name,
            grade_strength=grade_strength,
            sky_protection=sky_protection,
            shadow_depth=shadow_depth,
            green_tone=green_tone,
            anti_cgi=anti_cgi,
            preserve_geometry=protect_architecture and protect_people and preserve_text,
        )
        st.session_state["realphoto_result"] = result

    if "realphoto_result" in st.session_state:
        result = st.session_state["realphoto_result"]
        with right:
            st.image(result, caption="REALPHOTO PRO", use_container_width=True)

        export = resize_long_edge(result, export_size)
        buf = io.BytesIO()
        if export_format == "PNG":
            export.save(buf, format="PNG", optimize=True)
            mime = "image/png"
            ext = "png"
        else:
            export.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
            mime = "image/jpeg"
            ext = "jpg"

        st.download_button(
            f"СКАЧАТЬ {export_size} · {export_format}",
            data=buf.getvalue(),
            file_name=f"realphoto_pro_v2.{ext}",
            mime=mime,
            use_container_width=True,
        )
