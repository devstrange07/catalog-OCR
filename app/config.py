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
    parse_channel_default: str = "paddleocr"
    ocr_lang: str = "en"
    ocr_use_gpu: bool = False
    ocr_gpu_id: int = 0
    host: str = "0.0.0.0"
    port: int = 8000
    max_extract_chars: int = 24000
    extract_pages_per_chunk: int = 2
    extract_max_tokens: int = 8192
    crawl_timeout_sec: int = 60
    crawl_headless: bool = True


settings = Settings()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
