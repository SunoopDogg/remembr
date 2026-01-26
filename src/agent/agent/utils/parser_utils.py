"""JSON parsing utility functions."""

import json
import re


def parse_json(string: str) -> dict:
    """Parse JSON from string, handling ```json``` code blocks."""
    if '```json' in string:
        match = re.search(r"```json(.*?)```", string, re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1).strip())
    return json.loads(string)
