from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
STATIC_DIR = ROOT_DIR / "static"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_inline_max_bytes: int = 2 * 1024 * 1024  # 2 MB
    parse_channel_default: str = "paddleocr"
    ocr_lang: str = "en"
    ocr_use_gpu: bool = False
    ocr_gpu_id: int = 0
    host: str = "0.0.0.0"
    port: int = 8000
    max_extract_chars: int = 24000
    extract_pages_per_chunk: int = 2
    extract_max_tokens: int = 32768
    langsmith_enabled: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "product-ocr"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tracing_v2: bool = True
    crawl_timeout_sec: int = 60
    crawl_headless: bool = True
    unlimited_ocr_model: str = "baidu/Unlimited-OCR"
    unlimited_ocr_gpu_id: int = 0
    unlimited_ocr_max_length: int = 32768
    unlimited_ocr_base_size: int = 1024
    unlimited_ocr_image_size: int = 640
    unlimited_ocr_no_repeat_ngram_size: int = 35
    unlimited_ocr_ngram_window: int = 128
    unlimited_ocr_remote_url: str = ""
    unlimited_ocr_remote_api_key: str = ""
    unlimited_ocr_remote_timeout_sec: int = 180


settings = Settings()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
