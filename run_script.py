#!/usr/bin/env python
"""
Standalone parameter-sweep runner for the floorplan-vectorization pipeline.

Loads preprocess/isolate_walls/detect_walls/regularize/compute_metrics
and log_performance/plot_history straight out of main.ipynb, then
loops over PARAM_GRID below: one dataset pass per entry, each logged with the
notebook's own log_performance()/plot_history(). PARAM_GRID currently holds
two runs -- BASELINE and FINAL (the empirical optimum from the
per-parameter sweeps already recorded in GRAPHS/) -- for a final head-to-head
comparison rather than another sweep.

"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: no GUI windows when running outside Jupyter

NOTEBOOK_PATH = "main.ipynb"
IMAGES_GT_DIR = "DATASETS/CVC-FP"

# Every run is a dict of overrides; "tag"/"description" are passed straight to
# log_performance(). Omitting a key falls back to that function's own default
# (the value evaluate_pair already scores), but BASELINE below pins every key
# explicitly so FINAL's changes are unambiguous one-at-a-time deltas.

# isolate_walls takes no tunable kwargs
# detect_walls (threshold_frac, minlen_frac, maxgap_frac, rho_px, theta_deg) -- the Hough params.
# regularize (angle_tolerance, merge_distance_frac, corner_snap_frac)
BASELINE = {
    "threshold_frac": 0.01,
    "minlen_frac": 0.01,
    "maxgap_frac": 0.005,
    "rho_px": 1,
    "theta_deg": 1,
    "angle_tolerance": 5,
    "merge_distance_frac": 0.015,
    "corner_snap_frac": 0.007,
    "tolerance_frac": 0.0025,
}

# Empirical optimum found by the per-parameter sweeps recorded in GRAPHS/
# (see the "Hyperparameter sensitivity" section in main.ipynb): one-at-a-time
# changes from BASELINE, kept only where a sweep showed a real (non-noise)
# gain. rho_px, minlen_frac, angle_tolerance are left at baseline since the
# sweeps found them already optimal / inert over the tested range.
# tolerance_frac is also left at baseline: it's the evaluation matching
# tolerance, not a pipeline parameter, so raising it inflates F1 without
# improving the actual geometry.
FINAL = dict(BASELINE)
FINAL.update({
    "threshold_frac": 0.02,
    "maxgap_frac": 0.0025,
    "merge_distance_frac": 0.01,
    "corner_snap_frac": 0.003,
})

PARAM_GRID = [
    {"tag": "baseline", "description": "baseline parameters", **BASELINE},
    {"tag": "final", "description": "final decision parameters", **FINAL},
]

DETECT_KEYS = ("threshold_frac", "minlen_frac", "maxgap_frac", "rho_px", "theta_deg")
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
    isolate_walls = ns["isolate_walls"]
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
        detect_kwargs = {k: run[k] for k in DETECT_KEYS if k in run}
        regularize_kwargs = {k: run[k] for k in REGULARIZE_KEYS if k in run}
        tolerance_frac = run.get("tolerance_frac", None)

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
            walls_mask = isolate_walls(binary_img)
            segments = detect_walls(walls_mask, **detect_kwargs)
            regularized = regularize(segments, (h, w), **regularize_kwargs)
            # let compute_metrics own the fraction-to-pixel conversion (falls back to
            # its own default tolerance_frac when this run doesn't override it)
            metric_kwargs = {} if tolerance_frac is None else {"tolerance_frac": tolerance_frac}
            metrics = compute_metrics(regularized, polygons, (h, w), **metric_kwargs)
            results.append(metrics)

        log_performance(results, tag=tag, description=description)
        plot_history()  # refresh the trend chart after every run, so progress is visible mid-sweep

    print("\nsweep complete.", flush=True)


if __name__ == "__main__":
    main()
