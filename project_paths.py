"""Paths relative to this project; importing this module does not create directories."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def project_path(relative):
    """Return a native absolute path, preserving a requested trailing separator."""
    import os
    path = str(PROJECT_ROOT / relative)
    return path + os.sep if relative.endswith('/') else path

def model_path(relative):
    """Shared location for freshly trained weights, preserving checkpoint names."""
    path = PROJECT_ROOT / 'model' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path) + ('/' if relative.endswith('/') else '')
