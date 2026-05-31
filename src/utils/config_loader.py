"""
Configuration loading utilities.

Supports YAML inheritance (`base`) and deep-merge override.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
import copy


def load_config(config_path: str) -> Dict:
    """
    Load a YAML config file with recursive `base` support.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load current config file.
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Merge inherited base configs if provided.
    if 'base' in config:
        base_configs = config.pop('base')
        if not isinstance(base_configs, list):
            base_configs = [base_configs]
        
        # Recursively load base configs.
        merged = {}
        for base_path in base_configs:
            # Resolve relative paths from current config directory.
            if not Path(base_path).is_absolute():
                base_path = config_path.parent / base_path
            
            base_config = load_config(str(base_path))
            merged = merge_configs(merged, base_config)
        
        # Merge current config on top of merged bases.
        config = merge_configs(merged, config)
    
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge nested dictionaries."""
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries.
            result[key] = merge_configs(result[key], value)
        else:
            # Override leaf values directly.
            result[key] = value
    
    return result


def apply_overrides(config: Dict, overrides: List[str]) -> Dict:
    """
    Apply dot-path overrides to a config dict.

    Each override has the form ``key.subkey=value``.  Values are
    auto-cast to int / float / bool / str.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set format (expected key=value): {item!r}")
        key_path, raw_value = item.split("=", 1)
        keys = key_path.split(".")
        value: Any = _auto_cast(raw_value)
        sub = config
        for k in keys[:-1]:
            sub = sub.setdefault(k, {})
        sub[keys[-1]] = value
    return config


def _auto_cast(raw: str) -> Any:
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("none", "null", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def save_config(config: Dict, path: str):
    """Save config dictionary to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"[INFO] Config saved to {path}")

