#!/usr/bin/env python3

import tarfile
import zipfile
import tempfile
import shutil
import os
import sys
import json
import subprocess
import urllib.request


def extract_zip(fobj, dest):
    with zipfile.ZipFile(fobj) as z:
        for i in z.infolist():
            if i.is_dir():
                continue
            parts = os.path.normpath(i.filename).split(os.sep)
            basedir = os.path.join(dest, *parts[1:-1])
            os.makedirs(basedir, exist_ok=True)
            with z.open(i.filename) as src, open(
                os.path.join(basedir, parts[-1]), "wb"
            ) as dst:
                shutil.copyfileobj(src, dst)


def extract_tgz(fobj, dest):
    with tarfile.open(fileobj=fobj, mode="r:gz") as t:
        for i in t:
            if i.isdir():
                continue
            parts = os.path.normpath(i.name).split(os.sep)
            basedir = os.path.join(dest, *parts[1:-1])
            os.makedirs(basedir, exist_ok=True)
            targetpath = os.path.join(basedir, parts[-1])
            if i.issym():
                if os.path.lexists(targetpath):
                    os.unlink(targetpath)
                os.symlink(i.linkname, targetpath)
            else:
                src = t.extractfile(i)
                assert src != None
                with open(targetpath, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                    os.chmod(dst.fileno(), i.mode)


def download_and_extract(url, dst):
    print("Downloading {}...".format(url))
    r = urllib.request.urlopen(url)
    with tempfile.TemporaryFile() as f:
        shutil.copyfileobj(r, f)
        f.seek(0)

        print("Extracting to {}...".format(dst))
        if url.endswith(".zip"):
            extract_zip(f, dst)
        elif url.endswith(".tgz"):
            extract_tgz(f, dst)


def host_target():
    rustc = subprocess.check_output(["rustc", "-V", "-v"]).decode("utf-8")
    for l in rustc.splitlines():
        if l.startswith("host"):
            return l.split()[-1]
    assert False, "cannot determine host target"


def onnxruntime_download_url(target, version="1.12.1"):
    assert target != None
    target = target.split("-")
    os = target[2]
    arch = target[0]

    base = f"https://github.com/microsoft/onnxruntime/releases/download/v{version}"

    if os == "linux":
        if arch == "x86_64":
            return f"{base}/onnxruntime-linux-x64-{version}.tgz"
        elif arch == "aarch64":
            return f"{base}/onnxruntime-linux-aarch64-{version}.tgz"
    elif os == "darwin":
        return f"{base}/onnxruntime-osx-universal2-{version}.tgz"
    elif os == "windows":
        if arch == "x86_64":
            return f"{base}/onnxruntime-win-x64-{version}.zip"
        elif arch == "aarch64":
            return f"{base}/onnxruntime-win-arm64-{version}.zip"
    assert False, f"Unsupported OS/architecture combination: {os}/{arch}"


if __name__ == "__main__":
    basedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "vendor")
    target = host_target()
    if len(sys.argv) > 1:
        target = os.path.normpath(sys.argv[1])
        if target.endswith(".prepare"):
            target = target.split(os.sep)[-2]
    basedir = os.path.join(basedir, target)
    url = onnxruntime_download_url(target=target)

    cachepath = os.path.join(basedir, ".prepare")
    cache = {}
    if os.path.exists(cachepath):
        with open(cachepath) as f:
            cache = json.load(f)
    if cache.get("url") != url:
        download_and_extract(url, basedir)
        cache["url"] = url
    with open(cachepath, "w") as f:
        json.dump(cache, f)
