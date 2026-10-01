# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Build the Windows version for students:   python build_windows.py

1. Panda3D's build_apps freezes the program into build/win_amd64/ (works on Windows, macOS or Linux)
2. adds the Microsoft C++ runtime DLLs that numpy / matplotlib need (VCRUNTIME140_1.dll ...), so the
   students do NOT have to install the "Visual C++ Redistributable"
3. zips it:  dist/LLMPipelineDemo-win64.zip   ->  unzip, double-click LLMPipelineDemo.exe

Needs: pip install panda3d   (and internet access, to download the Windows wheels)
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build", "win_amd64")
DIST = os.path.join(HERE, "dist")
RUNTIME_DLLS = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"]

HOW_TO = """LLM Pipeline Demo  -  how to run
=================================

1. Unzip the whole folder (do not run it from inside the zip).
2. Double-click  LLMPipelineDemo.exe
   (If Windows SmartScreen says "Windows protected your PC": click "More info" -> "Run anyway".)

Keys:  Space = next step   A = auto play   1-7 = pick a question   D = deep dive on/off
       R = restart   H = hide panels   right-drag / wheel = rotate / zoom

Charts and pictures made by the demo are saved in the "outputs" folder next to the .exe.
If the program does not start, send the teacher this file:
    %APPDATA%\\LLMPipelineDemo\\output.log
"""


def run_build_apps():
    shutil.rmtree(BUILD, ignore_errors=True)
    subprocess.check_call([sys.executable, "setup.py", "build_apps"], cwd=HERE)


def add_runtime():
    """App-local copies of the Visual C++ runtime (allowed by Microsoft's redistribution terms)."""
    tmp = tempfile.mkdtemp()
    subprocess.check_call([sys.executable, "-m", "pip", "download", "msvc-runtime", "--platform", "win_amd64",
                           "--only-binary=:all:", "--no-deps", "--python-version", "3.11", "-d", tmp])
    whl = glob.glob(os.path.join(tmp, "msvc_runtime-*.whl"))[0]
    existing = {f.lower(): f for f in os.listdir(BUILD)}
    with zipfile.ZipFile(whl) as z:
        for name in z.namelist():
            base = os.path.basename(name).lower()
            if base in RUNTIME_DLLS and "/Scripts/" in name:
                target = existing.get(base, base)            # keep the existing spelling (VCRUNTIME140.dll)
                with z.open(name) as src, open(os.path.join(BUILD, target), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print("added", target)
    shutil.rmtree(tmp, ignore_errors=True)


def make_zip():
    with open(os.path.join(BUILD, "HOW TO RUN.txt"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(HOW_TO)
    os.makedirs(os.path.join(BUILD, "outputs"), exist_ok=True)
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, "LLMPipelineDemo-win64.zip")
    if os.path.exists(out):
        os.remove(out)
    top = "LLMPipelineDemo"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, dirs, files in os.walk(BUILD):
            rel = os.path.relpath(root, BUILD)
            if not files and not dirs:
                z.writestr(os.path.join(top, rel).replace("\\", "/") + "/", "")
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.join(top, os.path.relpath(p, BUILD)))
    print("wrote", out, "({:.0f} MB)".format(os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    run_build_apps()
    add_runtime()
    make_zip()
