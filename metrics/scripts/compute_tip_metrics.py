#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

METRIC_ROOT = Path("/mnt/data_disk/zyh/project/TM/PUCN_learned")
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from NLPD import NLPD_Loss
from standard_luminance import read_hdr_rgb_numpy, rgb_to_luminance_numpy
from tmqi import TMQI
from tmqi2 import TMQI2


LDR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRICS = [
    ("tmqi_q", "TMQI-Q"),
    ("tmqi_s", "TMQI-S"),
    ("tmqi_n", "TMQI-N"),
    ("tmqi2_q", "TMQI2-Q"),
    ("tmqi2_s", "TMQI2-S"),
    ("tmqi2_n", "TMQI2-N"),
    ("nlpd", "NLPD"),
]


def image_id(path):
    match = re.match(r"^(a\d+)", Path(path).stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def gather_last_hdrs(hdr_dir, count):
    files = sorted(Path(hdr_dir).rglob("*.exr"))
    if len(files) < count:
        raise RuntimeError(f"Found only {len(files)} EXR files, need {count}")
    selected = files[-count:]
    ids = [image_id(path) for path in selected]
    if any(value is None for value in ids):
        raise RuntimeError("Some reference filenames do not start with an aNNNN id")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate aNNNN ids found in selected references")
    return selected


def build_ldr_index(ldr_dir):
    index = {}
    duplicates = {}
    for path in sorted(Path(ldr_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in LDR_EXTENSIONS:
            continue
        key = image_id(path)
        if key is None:
            continue
        if key in index:
            duplicates.setdefault(key, [str(index[key])]).append(str(path))
            continue
        index[key] = path
    return index, duplicates


def read_ldr_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read LDR image: {path}")
    if image.ndim == 2:
        rgb = np.repeat(image[:, :, None], 3, axis=2)
    else:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    if rgb.dtype == np.uint8:
        result = rgb.astype(np.float32) / 255.0
    elif rgb.dtype == np.uint16:
        result = rgb.astype(np.float32) / 65535.0
    else:
        result = rgb.astype(np.float32)
        if np.nanmax(result) > 1.5:
            result /= 255.0
    return np.clip(result, 0.0, 1.0)


def resize_longest(image, longest):
    height, width = image.shape[:2]
    current = max(height, width)
    if longest <= 0 or current <= longest:
        return image
    scale = longest / float(current)
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else math.nan


def mean_finite(rows, field):
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return float(np.mean(values)) if values else math.nan


def compute_one(hdr_path, ldr_path, metric_resize, nlpd_metric, device):
    # 1000HDR/HDRtest EXRs are already linear absolute/relative HDR values.
    hdr = read_hdr_rgb_numpy(
        str(hdr_path), transfer="linear", primaries="rec709"
    ).astype(np.float32)
    hdr = np.clip(hdr, 0.0, None)
    ldr = read_ldr_rgb(ldr_path)

    if ldr.shape[:2] != hdr.shape[:2]:
        ldr = cv2.resize(
            ldr, (hdr.shape[1], hdr.shape[0]), interpolation=cv2.INTER_AREA
        )

    hdr_eval = resize_longest(hdr, metric_resize)
    ldr_eval = resize_longest(ldr, metric_resize)
    if ldr_eval.shape[:2] != hdr_eval.shape[:2]:
        ldr_eval = cv2.resize(
            ldr_eval,
            (hdr_eval.shape[1], hdr_eval.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    hdr_lum = rgb_to_luminance_numpy(hdr_eval, primaries="rec709")
    ldr_lum = rgb_to_luminance_numpy(ldr_eval, primaries="rec709")
    hdr_tensor = torch.from_numpy(hdr_lum).float().unsqueeze(0).unsqueeze(0).to(device)
    ldr_tensor = torch.from_numpy(ldr_lum).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        nlpd = nlpd_metric(
            torch.clamp(hdr_tensor, min=1e-6),
            torch.clamp(ldr_tensor, min=1e-6),
        ).item()

    ldr_u8 = np.clip(ldr_eval * 255.0, 0, 255).astype(np.uint8)
    q1, s1, n1, _, _ = TMQI(hdr_eval, ldr_u8)
    q2, s2, n2, _ = TMQI2(hdr_eval, ldr_u8)
    return {
        "tmqi_q": finite(q1),
        "tmqi_s": finite(s1),
        "tmqi_n": finite(n1),
        "tmqi2_q": finite(q2),
        "tmqi2_s": finite(s2),
        "tmqi2_n": finite(n2),
        "nlpd": finite(nlpd),
        "height": int(hdr_eval.shape[0]),
        "width": int(hdr_eval.shape[1]),
    }


def compute(args):
    references = gather_last_hdrs(args.hdr_dir, args.last_count)
    ldr_index, duplicates = build_ldr_index(args.ldr_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    nlpd_metric = NLPD_Loss().to(device).eval()

    rows = []
    missing = []
    errors = []
    for number, hdr_path in enumerate(references, start=1):
        key = image_id(hdr_path)
        ldr_path = ldr_index.get(key)
        if ldr_path is None:
            missing.append({"id": key, "hdr_path": str(hdr_path)})
            print(f"[{number}/{len(references)}] {key}: missing LDR", flush=True)
            continue
        try:
            values = compute_one(
                hdr_path, ldr_path, args.metric_resize_size, nlpd_metric, device
            )
            row = {
                "id": key,
                "hdr_path": str(hdr_path),
                "ldr_path": str(ldr_path),
                **values,
            }
            rows.append(row)
            print(
                f"[{number}/{len(references)}] {key}: "
                f"TMQI={row['tmqi_q']:.4f}, "
                f"TMQI2={row['tmqi2_q']:.4f}, NLPD={row['nlpd']:.4f}",
                flush=True,
            )
        except Exception as exc:
            errors.append({
                "id": key,
                "hdr_path": str(hdr_path),
                "ldr_path": str(ldr_path),
                "error": str(exc),
            })
            print(f"[{number}/{len(references)}] {key}: error: {exc}", flush=True)

    fields = [
        "id", "hdr_path", "ldr_path", "height", "width",
        *[field for field, _ in METRICS],
    ]
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "group": args.group,
        "method": args.method,
        "hdr_dir": str(Path(args.hdr_dir).resolve()),
        "ldr_dir": str(Path(args.ldr_dir).resolve()),
        "selection": f"last {args.last_count} references after filename sort",
        "first_reference": references[0].name,
        "last_reference": references[-1].name,
        "num_expected": len(references),
        "num_valid": len(rows),
        "num_missing": len(missing),
        "num_errors": len(errors),
        "num_duplicate_output_ids": len(duplicates),
        **{
            f"mean_{field}": mean_finite(rows, field)
            for field, _ in METRICS
        },
        "missing": missing,
        "errors": errors,
        "duplicate_output_ids": duplicates,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (output_dir / "summary.txt").open("w") as handle:
        for key, value in summary.items():
            if key in {"missing", "errors", "duplicate_output_ids"}:
                continue
            if isinstance(value, float):
                value = f"{value:.6f}" if math.isfinite(value) else "nan"
            handle.write(f"{key}: {value}\n")

    print(json.dumps({
        key: value for key, value in summary.items()
        if key not in {"missing", "errors", "duplicate_output_ids"}
    }, indent=2, ensure_ascii=False))
    return 0 if rows else 1


def display(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "--"


def latex_escape(text):
    mapping = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "(": r"(",
        ")": r")",
    }
    return "".join(mapping.get(char, char) for char in text)


def report(args):
    result_root = Path(args.result_root)
    summaries = []
    for path in sorted(result_root.glob("*/*/summary.json")):
        try:
            summaries.append(json.loads(path.read_text()))
        except Exception as exc:
            print(f"[!] Cannot read {path}: {exc}")
    summaries.sort(key=lambda item: (item["group"], item["method"].lower()))
    if not summaries:
        raise RuntimeError(f"No summary.json files under {result_root}")

    flat_path = result_root / "paper_summary.csv"
    fields = [
        "group", "method", *[name for _, name in METRICS],
        "valid", "expected", "missing", "errors",
    ]
    with flat_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for item in summaries:
            writer.writerow([
                item["group"],
                item["method"],
                *[display(item.get(f"mean_{field}", math.nan)) for field, _ in METRICS],
                item["num_valid"],
                item["num_expected"],
                item["num_missing"],
                item["num_errors"],
            ])

    grouped_path = result_root / "paper_summary_grouped.csv"
    with grouped_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Group", "TMO Method", "1000HDR Test (Last 400)"]
            + [""] * (len(METRICS) - 1)
        )
        writer.writerow(["", ""] + [name for _, name in METRICS])
        for item in summaries:
            writer.writerow([
                item["group"],
                item["method"],
                *[display(item.get(f"mean_{field}", math.nan)) for field, _ in METRICS],
            ])

    markdown = [
        "# TIP TMO Metric Report",
        "",
        "Dataset: 1000HDR HDRtest, filename-sorted last 400 references.",
        "",
        "| Group | TMO Method | "
        + " | ".join(name for _, name in METRICS)
        + " | Valid/Expected | Missing | Errors |",
        "| --- | --- | "
        + " | ".join(["---:"] * len(METRICS))
        + " | ---: | ---: | ---: |",
    ]
    for item in summaries:
        values = [
            display(item.get(f"mean_{field}", math.nan))
            for field, _ in METRICS
        ]
        markdown.append(
            f"| {item['group']} | {item['method']} | "
            + " | ".join(values)
            + f" | {item['num_valid']}/{item['num_expected']} "
            + f"| {item['num_missing']} | {item['num_errors']} |"
        )
    (result_root / "paper_report.md").write_text("\n".join(markdown) + "\n")

    columns = "ll" + "c" * len(METRICS)
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        rf"Group & TMO Method & \multicolumn{{{len(METRICS)}}}{{c}}{{1000HDR Test (Last 400)}} \\",
        " &  & " + " & ".join(name for _, name in METRICS) + r" \\",
        r"\midrule",
    ]
    previous_group = None
    for item in summaries:
        if previous_group is not None and item["group"] != previous_group:
            latex.append(r"\midrule")
        values = [
            display(item.get(f"mean_{field}", math.nan))
            for field, _ in METRICS
        ]
        latex.append(
            f"{latex_escape(item['group'])} & {latex_escape(item['method'])} & "
            + " & ".join(values)
            + r" \\"
        )
        previous_group = item["group"]
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Tone-mapping quality on the last 400 images of 1000HDR HDRtest. "
        r"Higher is better for TMQI/TMQI2; lower is better for NLPD.}",
        r"\label{tab:tip_tmo_1000hdr}",
        r"\end{table*}",
    ])
    (result_root / "paper_table.tex").write_text("\n".join(latex) + "\n")
    print(f"[*] Wrote {flat_path}")
    print(f"[*] Wrote {grouped_path}")
    print(f"[*] Wrote {result_root / 'paper_report.md'}")
    print(f"[*] Wrote {result_root / 'paper_table.tex'}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute_parser = subparsers.add_parser("compute")
    compute_parser.add_argument("--hdr_dir", required=True)
    compute_parser.add_argument("--ldr_dir", required=True)
    compute_parser.add_argument("--output_dir", required=True)
    compute_parser.add_argument("--group", required=True)
    compute_parser.add_argument("--method", required=True)
    compute_parser.add_argument("--last_count", type=int, default=400)
    compute_parser.add_argument("--metric_resize_size", type=int, default=512)
    compute_parser.add_argument("--device", default="cuda")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--result_root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(compute(arguments) if arguments.command == "compute" else report(arguments))
