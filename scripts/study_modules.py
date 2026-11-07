"""Import a study-specific tool without treating its stable ID as a Python identifier."""
from importlib import import_module


def load(study_id, name):
    return import_module(f"studies.{study_id}.scripts.{name}")
