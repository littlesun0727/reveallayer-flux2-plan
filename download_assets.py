"""Download the pinned first batch of models/data, with resumable Hub downloads."""

import argparse
import hashlib
import json
from pathlib import Path


def digest_file(path, algorithm):
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest.update("blob {}\0".format(path.stat().st_size).encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Asset storage directory")
    parser.add_argument("--component", choices=("all", "data", "klein", "xvae"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print pinned files; no network or writes")
    parser.add_argument("--verify-hashes", action="store_true", help="Also verify upstream hashes after download")
    parser.add_argument("--verify-only", action="store_true", help="Verify local files/hashes without network")
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name("configs").joinpath("assets.lock.json").read_text())
    components = lock["components"]
    keys = ("data", "xvae", "klein") if args.component == "all" else (args.component,)
    root = args.root.resolve()
    total = sum(f["size"] for key in keys for f in components[key]["files"])
    print("Selected files: {:.2f} GB total, including already downloaded files.".format(total / 1e9), flush=True)
    report = {"schema_version": 1, "components": {}, "passed": False}
    for key in keys:
        item = components[key]
        target = root / item["directory"]
        print("{} @ {} -> {}".format(item["repo_id"], item["revision"], target), flush=True)
        if args.dry_run:
            for file in item["files"]:
                print("  {} ({:,} bytes)".format(file["path"], file["size"]))
            continue
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=item["repo_id"], repo_type=item["repo_type"], revision=item["revision"],
                local_dir=str(target), allow_patterns=[f["path"] for f in item["files"]], max_workers=4,
            )
        verified = []
        for file in item["files"]:
            path = target / file["path"]
            if not path.is_file() or path.stat().st_size != file["size"]:
                raise RuntimeError("Missing or incorrect size: {}".format(path))
            checked_hash = None
            if args.verify_hashes or args.verify_only:
                algorithm = "sha256" if file["sha256"] else "sha1"
                expected = file["sha256"] or file["git_blob_sha1"]
                checked_hash = digest_file(path, algorithm)
                if checked_hash != expected:
                    raise RuntimeError("Hash mismatch: {}".format(path))
            verified.append({"path": file["path"], "size": file["size"], "hash_verified": checked_hash is not None})
        report["components"][key] = {"revision": item["revision"], "files": verified}
        print("{}: file checks passed.".format(key), flush=True)
    if not args.dry_run:
        report["passed"] = True
        destination = root / "manifests" / ("assets_{}.json".format(args.component))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Saved: {}".format(destination))


if __name__ == "__main__":
    main()
