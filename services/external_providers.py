import os

def external_provider_status():
    return {
        "gpt_image": bool(os.getenv("OPENAI_API_KEY")),
        "nano_banana": bool(os.getenv("GEMINI_API_KEY")),
        "magnific": bool(os.getenv("MAGNIFIC_API_KEY")),
    }

def run_external_provider(image, provider, prompt):
    """
    Integration point intentionally left provider-agnostic.
    Tech team should implement concrete SDK/HTTP calls here.
    The Streamlit UI and secret detection are already wired.
    """
    status = external_provider_status()

    if provider == "GPT Image":
        if not status["gpt_image"]:
            return None, "GPT Image: OPENAI_API_KEY не настроен."
        return None, "GPT Image: ключ найден. Реализуйте image-edit вызов в services/external_providers.py."

    if provider == "Nano Banana":
        if not status["nano_banana"]:
            return None, "Nano Banana: GEMINI_API_KEY не настроен."
        return None, "Nano Banana: ключ найден. Реализуйте Gemini image-edit вызов в services/external_providers.py."

    if provider == "Magnific":
        if not status["magnific"]:
            return None, "Magnific: MAGNIFIC_API_KEY не настроен."
        return None, "Magnific: ключ найден. Реализуйте API-клиент по вашему контракту Magnific."

    return None, "Неизвестный провайдер."
