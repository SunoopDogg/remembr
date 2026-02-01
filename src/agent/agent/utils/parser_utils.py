"""JSON parsing utility functions."""

import json
import re

_JSON_BLOCK_PATTERN = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json(string: str) -> dict:
    """Parse JSON from string, handling ```json``` code blocks.

    Falls back to ast.literal_eval when json.loads fails, matching
    the reference remembr behavior where eval() was used as fallback.
    """
    import ast

    content = string
    if '```json' in string:
        match = _JSON_BLOCK_PATTERN.search(string)
        if match:
            content = match.group(1).strip()

    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return ast.literal_eval(content)
