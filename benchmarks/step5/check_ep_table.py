"""Validate the effective EP16 table before loading the Step5 benchmark model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

KEY = "E=22,N=3072,K=4096,topk=8"


def check_ep_table(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = path.read_bytes()
    table = json.loads(raw)
    entries = table.get(KEY)
    if not isinstance(entries, dict) or not entries:
        raise ValueError(
            f"Step5 EP16 requires {KEY} in {path}; the effective table would "
            "otherwise fall back to untuned library configurations"
        )
    points = {}
    for size, pair in entries.items():
        try:
            size = int(size)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Step5 EP16 token count: {size!r}") from error
        if size <= 0 or size in points or not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid Step5 EP16 GEMM pair at M={size}")
        for config in pair:
            if not isinstance(config, dict) or any(
                type(config.get(name)) is not int or config[name] <= 0
                for name in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
            ):
                raise ValueError(f"Invalid Step5 EP16 GEMM dimensions at M={size}")
        if pair[0]["BLOCK_SIZE_M"] != pair[1]["BLOCK_SIZE_M"]:
            raise ValueError(f"Step5 EP16 GEMM pair has different BLOCK_SIZE_M at M={size}")
        points[size] = pair
    selected = {}
    for size in (256, 32768):
        nearest = min(points, key=lambda candidate: abs(candidate-size))
        selected[str(size)] = {"table_m": nearest, "gemm1": points[nearest][0], "gemm2": points[nearest][1]}
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "key": KEY, "entries": len(points), "selected": selected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        print("STEP5_EP16_TABLE " + json.dumps(check_ep_table(args.path)))
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")
