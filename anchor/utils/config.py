import json
from pathlib import Path

# Define the config path in the user's home directory (~/.anchor/config.json)
CONFIG_DIR = Path.home() / ".anchor"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "credentials": {
        "opensubtitles_api_key": "",
        "opensubtitles_username": "",
        "opensubtitles_password": "",
        "subdl_api_key": "",
    },
    "subtitle_preferences": {
        "subtitle_languages": ["en"],
        "prefer_sdh": False,
        "prefer_forced": False,
        "enable_podnapisi": False,
    },
    "hardware_overrides": {
        "audio_model": None,
        "batch_size": None,
        "translation_model": None
    }
}

def load_config() -> dict:
    """Loads the config file. Creates it with defaults if it doesn't exist."""
    if not CONFIG_FILE.exists():
        _create_default_config()
        return DEFAULT_CONFIG.copy()
        
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            
        # 1. Perform the merge in memory
        merged_config = _merge_configs(DEFAULT_CONFIG, user_config)
        
        # 2. Check if the merge actually added new missing keys
        if merged_config != user_config:
            # 3. Save the newly updated structure back to the file!
            save_config(merged_config)
            
        return merged_config
        
    except json.JSONDecodeError:
        # If the file got corrupted, return the defaults
        return DEFAULT_CONFIG.copy()

def save_config(config_data: dict):
    """Saves the configuration dictionary to the JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)

def _create_default_config():
    """Generates the initial config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    save_config(DEFAULT_CONFIG)

def _merge_configs(default: dict, user: dict) -> dict:
    """Recursively merges user config with defaults to ensure missing keys are added."""
    merged = default.copy()
    for key, value in user.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value     
    return merged