"""Inventory complete scenes and inspect a few images directly inside the first tar.

Does not extract the tar. This is a preliminary check, not the full 100-scene audit.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

from PIL import Image


def image_info(archive, member):
    with archive.extractfile(member) as stream, Image.open(stream) as image:
        image.load()
        result = {"mode": image.mode, "size": list(image.size)}
        if "A" in image.getbands():
            histogram = image.getchannel("A").histogram()
            pixels = sum(histogram)
            result["alpha_extrema"] = list(image.getchannel("A").getextrema())
            result["soft_alpha_fraction"] = sum(histogram[1:255]) / pixels
            result["nonzero_alpha_fraction"] = 1 - histogram[0] / pixels
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Same root as download_assets.py")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("checks/first_shard.json"))
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    raw = args.root / "data/raw/train"
    metadata = json.loads((raw / "trainMetaData.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError("Expected metadata to be a list of scene dictionaries.")
    report = {
        "scope": "preliminary pairing and image inspection; layer order and recomposition not verified",
        "metadata_scenes": len(metadata), "complete_scenes": 0,
        "partial_scenes": 0, "absent_scenes": 0, "invalid_metadata_scenes": 0,
        "inspected": [], "errors": [],
    }
    with tarfile.open(raw / "train-000000.tar", mode="r:") as archive:
        members = {}
        ambiguous = set()
        file_count = 0
        for member in archive.getmembers():
            if not member.isfile():
                continue
            file_count += 1
            # Upstream metadata uses scene_id/file.png. Permit a tar root prefix,
            # but reject duplicate suffixes instead of silently choosing one.
            key = "/".join(PurePosixPath(member.name).parts[-2:])
            if key in members:
                ambiguous.add(key)
            members[key] = member
        report["tar_regular_files"] = file_count
        report["ambiguous_member_paths"] = len(ambiguous)
        layer_counts = Counter()
        for scene in metadata:
            try:
                paths = [scene["full_image"], scene["background"], *scene["LayerInfoRaw"]]
                if len(paths) < 3 or any(not isinstance(path, str) or not path for path in paths):
                    raise ValueError("Missing required scene paths")
                keys = ["/".join(PurePosixPath(path).parts[-2:]) for path in paths]
            except (KeyError, TypeError, ValueError):
                report["invalid_metadata_scenes"] += 1
                continue
            found = [key in members and key not in ambiguous for key in keys]
            if not all(found):
                report["partial_scenes" if any(found) else "absent_scenes"] += 1
                continue
            report["complete_scenes"] += 1
            layer_counts[len(paths) - 2] += 1
            if len(report["inspected"]) >= args.samples:
                continue
            record = {"imgid": scene.get("imgid"), "issues": []}
            report["inspected"].append(record)
            try:
                images = [image_info(archive, members[key]) for key in keys]
                record["full_image"], record["background"] = images[:2]
                record["layers"] = images[2:]
                width, height = images[0]["size"]
                boxes = [detection["bbox"] for detection in scene["detections"]]
                record["boxes"] = boxes
                if len(boxes) != len(images) - 2:
                    record["issues"].append("box_layer_count_mismatch")
                for box in boxes:
                    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                        record["issues"].append("invalid_box")
                        continue
                    x1, y1, x2, y2 = box
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        record["issues"].append("box_outside_image_or_empty")
                if images[1]["size"] != images[0]["size"]:
                    record["issues"].append("background_size_differs")
                for layer in images[2:]:
                    layer["full_canvas"] = layer["size"] == images[0]["size"]
                    if layer["mode"] != "RGBA":
                        record["issues"].append("foreground_not_rgba")
                    if layer.get("nonzero_alpha_fraction") == 0:
                        record["issues"].append("empty_alpha")
                    if not layer["full_canvas"]:
                        record["issues"].append("cropped_foreground_needs_coordinate_mapping")
            except Exception as error:
                record["error"] = str(error)
                report["errors"].append({"imgid": scene.get("imgid"), "error": str(error)})
        report["complete_scene_layer_counts"] = dict(sorted(layer_counts.items()))
    report["preliminary_passed"] = (
        len(report["inspected"]) == args.samples and not report["errors"]
        and not report["ambiguous_member_paths"]
        and all(not record["issues"] for record in report["inspected"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "inspected"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved: {}".format(args.output.resolve()))
    return 0 if report["preliminary_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
