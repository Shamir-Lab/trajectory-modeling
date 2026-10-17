# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/gen_s1.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Blocks of Supplementary Table S1 (LaTeX, same format as the supplement) from the full-precision fold
scores used for the boxplots. usage: gen_s1.py <poster_root> <out_json>"""
import sys, json, glob
import pandas as pd
BS = chr(92)
MOD = {"logisticregression": "LR", "randomforest": "RF", "histgradientboosting": "HGB", "gradientboosting": "HGB", "xgboost": "XGB"}
FS = [("sd", "SD"), ("sd_sum", "SD+S"), ("sd_traj", "SD+T"), ("sd_sum_traj", "SD+S+T"), ("sd_traj_boot", "SD+T (B)"), ("sd_sum_traj_boot", "SD+S+T (B)")]
BLOCKS = [("liver", "mimic", "ACLF (MIMIC)"), ("liver", "eicu", "ACLF (eICU)"), ("aki", "mimic", "AKI-III (MIMIC)"),
          ("aki", "eicu", "AKI-III (eICU)"), ("circulatory_failure", "mimic", "CF (MIMIC)"), ("circulatory_failure", "eicu", "CF (eICU)"),
          ("circulatory_failure", "hirid", "CF (HiRID)"), ("sepsis", "mimic", "Sepsis (MIMIC)"), ("sepsis", "eicu", "Sepsis (eICU)"),
          ("ventilator", "mimic", "Vent. (MIMIC)"), ("ventilator", "eicu", "Vent. (eICU)")]
out = {}
for task, ds, label in BLOCKS:
    f = glob.glob(f"{sys.argv[1]}/{ds}/{task}/*/fold_metrics_selected_models.csv")
    assert len(f) == 1, (ds, task, f)
    d = pd.read_csv(f[0])
    d["mk"] = d.model.astype(str).str.lower().str.replace(" ", "").map(MOD)
    mean = d.groupby(["canonical_feature_set", "mk"])[["auroc", "aupr"]].mean()
    assert d.groupby(["canonical_feature_set", "mk"]).size().eq(50).all()
    cols = [(mk, met) for mk in ("LR", "RF", "HGB", "XGB") for met in ("auroc", "aupr")]
    val = {(fs, mk, met): float(mean.loc[(fs, mk), met]) for fs, _ in FS for mk, met in cols}
    best = {c: max((val[fs, c[0], c[1]], fs) for fs, _ in FS)[1] for c in cols}
    lines = []
    for i, (fs, fl) in enumerate(FS):
        cells = [(BS + "textbf{" + f"{val[fs, c[0], c[1]]:.3f}" + "}") if best[c] == fs else f"{val[fs, c[0], c[1]]:.3f}" for c in cols]
        head = (BS + "multirow{6}{*}{" + BS + "textbf{" + label + "}}") if i == 0 else "~"
        lines.append(f"{head} & {fl} & " + " & ".join(cells) + " " + BS + BS)
    out[label] = "\n".join(lines)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(len(out), "blocks")
