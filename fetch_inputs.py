"""Install published scientific inputs; no Torch, model execution or PF solves."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import uuid


HERE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError("Unsafe transport path")
    target = root.joinpath(*name.parts)
    if root not in target.resolve().parents:
        raise ValueError("Transport path escapes the selected workspace")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="A writable data-volume directory, never the almost-full system disk")
    parser.add_argument("--manifest", default="scientific_inputs.json",
                        choices=("scientific_inputs.json", "additional_inputs.json",
                                 "report_inputs.json", "company_reports_case118.json",
                                 "company_reports_case500.json"))
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(exist_ok=True)
    manifest_path = HERE / args.manifest
    manifest = json.loads(manifest_path.read_bytes())
    marker = root / (manifest_path.stem.upper() + "_INSTALLED.json")
    if marker.exists():
        if marker.read_bytes() != manifest_path.read_bytes():
            raise ValueError("This directory belongs to another input release")
        print("Already installed:", root)
        return
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise OSError("Use a writable data-volume directory with at least 20 GiB free")
    cache = root / "downloads" / manifest_path.stem
    cache.mkdir(parents=True, exist_ok=True)
    package = root / "science"
    package.mkdir(exist_ok=True)

    def download(part):
        target = cache / part["name"]
        if target.exists():
            if target.stat().st_size != part["bytes"] or sha(target) != part["sha256"]:
                raise ValueError("Existing complete download differs: " + target.name)
            return
        temporary = cache / (part["name"] + "." + uuid.uuid4().hex + ".partial")
        h = hashlib.sha256()
        request = urllib.request.Request(part["url"], headers={"User-Agent": "tcpa-input-download"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("xb") as output:
            for block in iter(lambda: response.read(1024**2), b""):
                h.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != part["bytes"] or h.hexdigest() != part["sha256"]:
            raise ValueError("Downloaded bytes differ; failed partial is preserved")
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
        print("Downloaded:", part["name"], flush=True)

    parts = [part for payload in manifest["payloads"] for part in payload["parts"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, parts))
    for index, payload in enumerate(manifest["payloads"]):
        target = (cache / f"payload{index:02d}.tar.gz" if payload["kind"] == "tar"
                  else safe_path(package, payload["destination"]))
        if target.exists():
            if sha(target) != payload["sha256"] or target.stat().st_size != payload["bytes"]:
                raise ValueError("Existing payload differs")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
            with temporary.open("xb") as output:
                for part in payload["parts"]:
                    with (cache / part["name"]).open("rb") as source:
                        shutil.copyfileobj(source, output, length=1024**2)
                output.flush()
                os.fsync(output.fileno())
            if sha(temporary) != payload["sha256"] or temporary.stat().st_size != payload["bytes"]:
                raise ValueError("Assembled payload differs")
            temporary.rename(target)
        if payload["kind"] == "tar" and not payload["role"].startswith("report_only"):
            with tarfile.open(target, "r:gz") as archive:
                for member in archive:
                    if not member.isfile():
                        raise ValueError("Only ordinary files are admitted")
                    destination = safe_path(package, member.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        with archive.extractfile(member) as source:
                            expected = hashlib.sha256(source.read()).hexdigest()
                        if sha(destination) != expected:
                            raise ValueError("Existing source or learning graph differs")
                    else:
                        with archive.extractfile(member) as source, destination.open("xb") as output:
                            shutil.copyfileobj(source, output, length=1024**2)
    with marker.open("xb") as output:
        output.write(manifest_path.read_bytes())
        output.flush()
        os.fsync(output.fileno())
    print("Actual source, N0 and training inputs installed:", package, flush=True)


if __name__ == "__main__":
    main()
