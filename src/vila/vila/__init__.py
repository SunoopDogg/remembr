import os
import sys


def _setup_llava_import_path() -> None:
    """Add vila package dir to sys.path for Hydra 'llava.*' imports."""
    vila_package_dir = os.path.dirname(__file__)
    if vila_package_dir not in sys.path:
        sys.path.insert(0, vila_package_dir)


_setup_llava_import_path()
