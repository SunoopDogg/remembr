"""File utility functions."""


def file_to_string(filepath: str) -> str:
    """Read file contents as string."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
