"""
Configuration module for loading and managing application settings.
Reads exclusively from config.toml and provides sensible defaults for missing keys.
"""

import tomllib
from pathlib import Path
from rich.console import Console
from i18n import init_i18n

console = Console()

# Define supported configuration keys explicitly
ALLOWED_KEYS = {
    "deepseek_api_key",
    "deepseek_model",
    "deepseek_fast_model",
    "deepseek_reasoning_model",
    "tesseract_cmd",
    "tesseract_lang",
    "log_language",
    "model_output_language",
    "formal_answer_timeout_seconds",
    "analysis_stage_mode",
    "single_stage_model",
    "request_retry_attempts",
}

# Defaults for all supported keys
DEFAULTS: dict = {
    "deepseek_api_key": "",
    "deepseek_model": "deepseek-reasoner",
    "deepseek_fast_model": "deepseek-chat",
    "deepseek_reasoning_model": "deepseek-reasoner",
    "tesseract_cmd": None,
    "tesseract_lang": "chi_sim+eng",
    "log_language": "zh",
    "model_output_language": None,
    "formal_answer_timeout_seconds": 60,
    "analysis_stage_mode": "multi",
    "single_stage_model": "deepseek-chat",
    "request_retry_attempts": 3,
}


def load_config() -> dict:
    """
    Load configuration from config.toml in the same directory.
    If a key is missing, fill it from DEFAULTS.

    Returns:
        dict: Configuration dictionary with loaded settings merged with defaults
    """
    base = Path(__file__).resolve().parent
    cfg_file = base / "config.toml"

    if not cfg_file.exists():
        i18n = init_i18n(DEFAULTS)
        console.print(i18n.t("config.not_found"))
        return dict(DEFAULTS)

    try:
        with cfg_file.open("rb") as f:
            loaded = tomllib.load(f) or {}
        # Filter only allowed keys
        filtered = {k: v for k, v in loaded.items() if k in ALLOWED_KEYS}
        # Merge with defaults (config overrides defaults)
        merged = {**DEFAULTS, **filtered}
        i18n = init_i18n(merged)
        console.print(i18n.t("config.loaded_success"))
        return merged
    except Exception as e:
        i18n = init_i18n(DEFAULTS)
        console.print(i18n.t("config.read_failed", path=str(cfg_file), error=str(e)))
        console.print(i18n.t("config.fallback_defaults"))
        return dict(DEFAULTS)


def get_config_value(config: dict, key: str, default=None):
    """
    Get configuration value from the loaded config, falling back to default.

    Args:
        config: Loaded configuration dictionary
        key: Configuration key to look up
        default: Default value if key is not present

    Returns:
        Any: Configuration value or provided default
    """
    return config.get(key, default)
