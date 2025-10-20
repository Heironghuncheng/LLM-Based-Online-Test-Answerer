"""
Simple i18n module with JSON locale catalogs.
Provides professional multi-language handling with ISO codes ('zh', 'en').
"""

import json
from pathlib import Path

LANG_ALIASES = {
    # Chinese aliases
    "chinese": "zh",
    "zh": "zh",
    "zh_cn": "zh",
    "zh-cn": "zh",
    "zh_hans": "zh",
    "zh-hans": "zh",
    "cn": "zh",
    "中文": "zh",
    # English aliases
    "english": "en",
    "en": "en",
    "en_us": "en",
    "en-us": "en",
    "英语": "en",
}


def normalize_lang_code(value: str | None, default: str = "zh") -> str:
    v = str(value).strip().lower() if value is not None else None
    code = LANG_ALIASES.get(v)
    if code in ("zh", "en"):
        return code
    return default


class I18n:
    def __init__(self, lang_code: str, fallback_lang_code: str = "en"):
        self.lang_code = lang_code
        self.fallback_lang_code = fallback_lang_code
        base = Path(__file__).resolve().parent / "locales"
        self.trans = self._load_json(base, lang_code)
        self.fallback_trans = self._load_json(base, fallback_lang_code)

    def _load_json(self, base: Path, code: str) -> dict:
        p = base / f"{code}.json"
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}
        return {}

    def t(self, key: str, **kwargs) -> str:
        text = self.trans.get(key) or self.fallback_trans.get(key) or key
        try:
            return text.format(**kwargs)
        except Exception:
            return text

    def name(self, code: str) -> str:
        # Return display name for a language code under current UI language
        if self.lang_code == "zh":
            return "中文" if code == "zh" else "英文"
        return "Chinese" if code == "zh" else "English"


def init_i18n(config: dict) -> I18n:
    # Config-only policy: read language from config, fallback to default
    cfg = (config or {}).get("log_language")
    lang_code = normalize_lang_code(cfg, default="zh")
    return I18n(lang_code)


def resolve_output_language(config: dict) -> str:
    # Config-only policy: read output language from config, fallback to UI language
    ui_lang = normalize_lang_code((config or {}).get("log_language"), default="zh")
    cfg = (config or {}).get("model_output_language")
    code = normalize_lang_code(cfg, default=ui_lang)
    return code