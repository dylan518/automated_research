import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


INCLUDE_PATTERNS = [
    "literature_search_agent.py",
    "semantic_scholar.py",
    "tree_of_thought.py",
    "README.md",
    "requirements.txt",
    "prompts/**/*.md",
]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def run_git(command: str, cwd: Path) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def collect_files(root: Path) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for pattern in INCLUDE_PATTERNS:
        for p in root.glob(pattern):
            if p.is_file():
                resolved = p.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(p)
    return sorted(files, key=lambda p: str(p))


def build_snapshot(root: Path, label: str) -> Dict:
    files = collect_files(root)
    file_entries = []
    aggregate_hasher = hashlib.sha256()

    for file_path in files:
        digest = sha256_file(file_path)
        rel = str(file_path.relative_to(root))
        file_entries.append(
            {
                "path": rel,
                "sha256": digest,
                "bytes": file_path.stat().st_size,
            }
        )
        aggregate_hasher.update(f"{rel}:{digest}".encode("utf-8"))

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    aggregate_digest = aggregate_hasher.hexdigest()
    snapshot_id = f"{now}-{aggregate_digest[:10]}"

    return {
        "snapshot_id": snapshot_id,
        "label": label,
        "created_at_utc": now,
        "git": {
            "head_short": run_git("git rev-parse --short HEAD", root),
            "status_porcelain": run_git("git status --short", root).splitlines(),
        },
        "file_count": len(file_entries),
        "aggregate_sha256": aggregate_digest,
        "files": file_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible code version snapshot")
    parser.add_argument(
        "--label",
        type=str,
        default="baseline",
        help="Human label for this code snapshot",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/version_refs",
        help="Directory to write snapshot json",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    snapshot = build_snapshot(root, args.label)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{snapshot['snapshot_id']}.json"
    out_file.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    print(f"snapshot_id={snapshot['snapshot_id']}")
    print(f"snapshot_file={out_file}")
    print(f"file_count={snapshot['file_count']}")


if __name__ == "__main__":
    main()
