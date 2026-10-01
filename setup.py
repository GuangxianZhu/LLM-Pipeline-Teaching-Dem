# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Package the demo as a stand-alone program (no Python needed on the students' computers).

    pip install panda3d
    python setup.py build_apps

Result (for Windows):  build/win_amd64/   ->  zip this folder, students unzip it and run LLMPipelineDemo.exe
This works from Windows, macOS or Linux (Panda3D downloads the Windows wheels by itself).
"""
from setuptools import setup

setup(
    name="LLMPipelineDemo",
    version="1.0.0",
    options={
        "build_apps": {
            "gui_apps": {"LLMPipelineDemo": "main.py"},
            "platforms": ["win_amd64"],
            "requirements_path": "build-requirements.txt",
            "use_optimized_wheels": False,          # plain PyPI wheels
            # data files that the program reads at run time (kept next to the .exe)
            "include_patterns": [
                "tiny/weights.npz",
                "tiny/vocab.json",
                "demo_files/*",
                "README.md",
            ],
            "exclude_patterns": ["legacy_factory/**", "outputs/**", "**/__pycache__/**"],
            "exclude_modules": {"*": ["torch", "tkinter", "IPython", "pytest", "setuptools", "pip"]},
            "plugins": ["pandagl", "p3openal_audio"],
            # errors are written here (the window has no console)
            "log_filename": "$USER_APPDATA/LLMPipelineDemo/output.log",
            "log_append": False,
        }
    },
)
