from __future__ import annotations

from pathlib import Path
import os
import shutil
import json


def main() -> None:
    scratch = Path(os.environ["SCRATCH"]) / "kg"
    canon = Path("/mnt/gcs/jouvencekb/main")
    promoted = []
    for kind in ["nodes", "edges"]:
        (canon / kind).mkdir(parents=True, exist_ok=True)
        for src in sorted((scratch / kind).glob("*.parquet")):
            dst = canon / kind / src.name
            if dst.exists():
                raise FileExistsError(f"canonical file already exists: {dst}")
            tmp = dst.with_suffix(dst.suffix + ".tmp-hermes-promote")
            if tmp.exists():
                tmp.unlink()
            print(f"[copy] {src} -> {dst}", flush=True)
            shutil.copy2(src, tmp)
            tmp.rename(dst)
            promoted.append({"kind": kind, "file": src.name, "bytes": dst.stat().st_size})
    print(json.dumps({"promoted": promoted}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
