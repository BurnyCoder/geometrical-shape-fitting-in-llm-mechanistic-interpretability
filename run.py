"""Single entry point: `uv run python run.py <phase> [<phase> ...] --config configs/<name>.yaml`.

Global context: each phase is one clearly named step of the research pipeline, implemented in a
module of src/neurogeom; this file only wires them together, sets up timestamped logging per phase,
and enforces the preregistration guard (no training before reports/preregistration.md is committed).
`all` runs the phases in order: check-env -> catalog -> train -> eval-synth -> extract-real ->
eval-real -> report -> gallery -> site (publish = HF upload, run explicitly).
See README.md ("What happens when you run it").
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurogeom.config import REPO_ROOT, load_config  # noqa: E402
from neurogeom.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger("run")


def prereg_committed() -> bool:
    """True if reports/preregistration.md is tracked by git with at least one commit."""
    try:
        out = subprocess.run(["git", "log", "--oneline", "--", "reports/preregistration.md"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return False


def phase_check_env(cfg):
    import torch
    log.info("python %s | torch %s | cuda %s | available %s", sys.version.split()[0], torch.__version__, torch.version.cuda,
             torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("gpu %s | arch list %s", torch.cuda.get_device_name(0), torch.cuda.get_arch_list())


def phase_catalog(cfg):
    from neurogeom.report.render import render_catalog
    render_catalog()


def phase_train(cfg):
    if not prereg_committed():
        raise SystemExit("refusing to train: commit reports/preregistration.md first")
    from neurogeom.train.loop import train
    train(cfg)


def phase_eval_synth(cfg):
    from neurogeom.eval.synth_eval import run_synth_eval
    run_synth_eval(cfg)


def phase_tune_b1_fixed(cfg):
    """Post hoc (audit 2026-11-14): fit B1-fixed's blob threshold tau on the validation stream."""
    import json
    from neurogeom.eval.synth_eval import run_b1_fixed, tune_blob_tau
    res = tune_blob_tau(run_b1_fixed(cfg, "val", tau=None))
    json.dump(res, open(Path(cfg["paths"]["results"]) / "synth" / "b1_fixed_tau.json", "w"), indent=2)
    log.info("B1-fixed tau = %.4f (val macro-F1 %.3f; never blob %.3f; %d val clouds)", res["tau"], res["val_macro_f1"],
             res["val_macro_f1_never_blob"], res["n_val"])


def phase_eval_b1_fixed(cfg):
    """Post hoc (audit 2026-11-14): B1-fixed on the baseline subset with the tuned tau; rows replace any earlier
    B1_fixed rows of per_sample.csv."""
    import json
    import pandas as pd
    from neurogeom.eval.synth_eval import run_b1_fixed
    synth = Path(cfg["paths"]["results"]) / "synth"
    tau = json.load(open(synth / "b1_fixed_tau.json"))["tau"]
    new = run_b1_fixed(cfg, "test", tau)
    old = pd.read_csv(synth / "per_sample.csv", low_memory=False)
    pd.concat([old[old.system != "B1_fixed"], new], ignore_index=True).to_csv(synth / "per_sample.csv", index=False)
    log.info("B1-fixed: %d rows, accuracy %.3f, blob predicted %d -> per_sample.csv", len(new), new.correct.mean(),
             int((new.family_pred == "blob").sum()))


def phase_extract_real(cfg):
    from neurogeom.realdata.extract import extract_all
    extract_all(cfg)


def phase_eval_real(cfg):
    from neurogeom.eval.real_eval import run_real_eval
    run_real_eval(cfg)


def phase_report(cfg):
    from neurogeom.report.numbers import build_all
    build_all(cfg)


def phase_gallery(cfg):
    import json
    import torch
    from neurogeom.eval.gallery import build_gallery
    from neurogeom.train.loop import final_checkpoint, load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(final_checkpoint(cfg), dev)
    t = json.load(open(Path(cfg["paths"]["results"]) / "synth" / "meta.json"))["temperature"]
    build_gallery(model, cfg, t, dev, Path(cfg["paths"]["results"]) / "gallery.json")


def phase_site(cfg):
    from neurogeom.report.site import build_site
    build_site(cfg, Path(cfg["paths"]["results"]) / "gallery.json")


def phase_publish(cfg):
    import os
    from neurogeom.report.publish import add_to_collection, upload_run
    repo = f"{os.environ.get('NEUROGEOM_HF_REPO', 'BurnyCoder/neurogeom-shape-fitter')}-{cfg['run_name'].replace('_', '-')}"
    upload_run(cfg, repo)
    add_to_collection([repo])


PHASES = {
    "check-env": phase_check_env,
    "catalog": phase_catalog,
    "train": phase_train,
    "eval-synth": phase_eval_synth,
    "tune-b1-fixed": phase_tune_b1_fixed,
    "eval-b1-fixed": phase_eval_b1_fixed,
    "extract-real": phase_extract_real,
    "eval-real": phase_eval_real,
    "report": phase_report,
    "gallery": phase_gallery,
    "site": phase_site,
    "publish": phase_publish,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="+", choices=list(PHASES) + ["all"])
    ap.add_argument("--config", default=None, help="YAML file in configs/ (merged over configs/base.yaml)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set eval.baselines=false (YAML-parsed value)")
    args = ap.parse_args(argv)
    overrides = {}
    import yaml
    for kv in args.set:
        key, val = kv.split("=", 1)
        node = overrides
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(val)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.run_name:
        overrides["run_name"] = args.run_name
    cfg = load_config(args.config, overrides)
    phases = [p for p in PHASES if p != "publish"] if "all" in args.phases else args.phases
    for ph in phases:
        log_path = setup_logging(cfg["paths"]["logs"], f"{cfg['run_name']}_{ph}")
        log.info("=== phase %s | config %s | run %s | log %s", ph, cfg["config_name"], cfg["run_name"], log_path)
        PHASES[ph](cfg)
        log.info("=== phase %s done", ph)


if __name__ == "__main__":
    main()
