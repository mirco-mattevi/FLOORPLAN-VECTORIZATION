#!/usr/bin/env python
"""
Standalone parameter-sweep runner for the floorplan-vectorization pipeline.

Loads preprocess/isolate_walls_multiscale/detect_walls/regularize/compute_metrics
and log_performance/plot_history straight out of main.ipynb, then
loops over PARAM_GRID below: one dataset pass per entry, each logged with the
notebook's own log_performance()/plot_history().

"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: no GUI windows when running outside Jupyter

NOTEBOOK_PATH = "main.ipynb"
IMAGES_GT_DIR = "DATASETS/CVC-FP"

# One dict per run. "tag"/"description" are passed straight to log_performance().
# isolate_walls_multiscale (thick_dist, thin_dist)
# detect_walls (threshold_frac, minlen_frac, maxgap_frac)
# regularize (angle_tolerance, merge_distance_frac, corner_snap_frac)
# compute_metrics (metric_tolerance)
# Omit a key to use that function's own default, i.e. the value evaluate_pair already scores.
PARAM_GRID = [
    
]

ISOLATE_KEYS = ("thick_dist", "thin_dist")
DETECT_KEYS = ("threshold_frac", "minlen_frac", "maxgap_frac")
REGULARIZE_KEYS = ("angle_tolerance", "merge_distance_frac", "corner_snap_frac")


def load_pipeline():
    """
    Exec every code cell of main.ipynb into a fresh namespace, so this script
    always runs the exact pipeline code the notebook currently defines.
    
    """
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    namespace = {"__name__": "__pipeline__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.strip().startswith("results = verify_dataset("):
            continue
        if "input(" in source:
            continue
        exec(compile(source, f"{NOTEBOOK_PATH}[cell {i}]", "exec"), namespace)
    return namespace


def main():
    ns = load_pipeline()
    cv2 = ns["cv2"]
    glob = ns["glob"]
    preprocess = ns["preprocess"]
    isolate_walls_multiscale = ns["isolate_walls_multiscale"]
    detect_walls = ns["detect_walls"]
    regularize = ns["regularize"]
    compute_metrics = ns["compute_metrics"]
    parse_gt_walls = ns["parse_gt_walls"]
    log_performance = ns["log_performance"]
    plot_history = ns["plot_history"]

    supported_formats = (".png", ".jpg", ".jpeg")
    pairs = []
    for filename in sorted(os.listdir(IMAGES_GT_DIR)):
        if not filename.lower().endswith(supported_formats):
            continue
        img_path = os.path.join(IMAGES_GT_DIR, filename)
        img_name = os.path.splitext(filename)[0]
        svg = glob.glob(os.path.join(IMAGES_GT_DIR, f"{img_name}_gt_*.svg"))
        if svg:
            pairs.append((img_path, svg[0]))
    print(f"found {len(pairs)} (image, GT) pairs in {IMAGES_GT_DIR}", flush=True)

    for run in PARAM_GRID:
        tag = run["tag"]
        description = run.get("description", "")
        isolate_kwargs = {k: run[k] for k in ISOLATE_KEYS if k in run}
        detect_kwargs = {k: run[k] for k in DETECT_KEYS if k in run}
        regularize_kwargs = {k: run[k] for k in REGULARIZE_KEYS if k in run}
        metric_tolerance = run.get("metric_tolerance", None)

        print(f"\n=== {tag} ===", flush=True)
        results = []
        for img_path, svg_path in pairs:
            img = cv2.imread(img_path)
            polygons, declared_size = parse_gt_walls(svg_path)
            h, w = img.shape[:2]
            if declared_size is not None and declared_size != (w, h):
                print(f"  skipping {img_path}: GT declares {declared_size}, image is {(w, h)}", flush=True)
                continue

            binary_img = preprocess(img)
            walls_mask = isolate_walls_multiscale(binary_img, **isolate_kwargs)
            segments = detect_walls(walls_mask, **detect_kwargs)
            regularized = regularize(segments, (h, w), **regularize_kwargs)
            metrics = compute_metrics(regularized, polygons, (h, w), tolerance=metric_tolerance)
            results.append(metrics)

        log_performance(results, tag=tag, description=description)
        plot_history()  # refresh the trend chart after every run, so progress is visible mid-sweep

    print("\nsweep complete.", flush=True)


if __name__ == "__main__":
    main()
