from typing import Any, Optional


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == 'null' or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any) -> Optional[str]:
    if value is None or value == 'null':
        return None
    return str(value) if value else None
