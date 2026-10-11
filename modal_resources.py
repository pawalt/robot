"""Shared deployment resources; no experiment behavior."""
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parent
volume = modal.Volume.from_name("robot-grid-recordings", create_if_missing=True)


def project_image(*, extras=()):
    return (modal.Image.debian_slim(python_version="3.14")
            .uv_sync(uv_project_dir=str(PROJECT), extras=list(extras))
            .add_local_file(PROJECT / "modal_resources.py", "/root/modal_resources.py"))
