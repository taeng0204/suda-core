"""Statistical significance tests for SUDA paper.

Computes Wilcoxon signed-rank tests and Cohen's d effect sizes
for Table 1 (SUDA vs baselines) and Table 3 (SUDA vs SUDA-NoUnlearn).
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std == 0:
        return float('inf') if np.mean(x) != np.mean(y) else 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def effect_size_label(d):
    """Interpret Cohen's d."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def wilcoxon_test(x, y, alternative='greater'):
    """Perform Wilcoxon signed-rank test (paired, one-sided by default)."""
    diff = np.array(x) - np.array(y)
    if np.all(diff == 0):
        return 1.0, 0.0  # p-value=1, statistic=0
    try:
        stat, p = stats.wilcoxon(diff, alternative=alternative)
        return p, stat
    except ValueError:
        # Too few samples or all zeros
        return 1.0, 0.0


def permutation_test(x, y, n_permutations=10000):
    """Exact permutation test for small samples."""
    x, y = np.array(x), np.array(y)
    observed_diff = np.mean(x) - np.mean(y)
    combined = np.concatenate([x, y])
    n = len(x)
    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(combined)
        perm_diff = np.mean(perm[:n]) - np.mean(perm[n:])
        if perm_diff >= observed_diff:
            count += 1
    return count / n_permutations


def main():
    results_dir = Path("note/260208/results")

    # Load raw results
    with open(results_dir / "table1_raw_results.json") as f:
        table1_raw = json.load(f)

    with open(results_dir / "table3_raw_results.json") as f:
        table3_raw = json.load(f)

    output = []
    output.append("=" * 70)
    output.append("STATISTICAL SIGNIFICANCE TESTS FOR SUDA PAPER")
    output.append("=" * 70)

    # ============================================================
    # TABLE 1: SUDA vs each baseline (G-mean)
    # ============================================================
    output.append("\n" + "=" * 70)
    output.append("TABLE 1: SUDA vs Baselines - G-mean Comparison")
    output.append("=" * 70)

    datasets = ["nslkdd", "unswnb15", "cicids2018"]
    baselines = ["ARF", "SRP", "LeveragingBagging", "HAT", "HoeffdingTree", "EFDT"]

    for dataset in datasets:
        output.append(f"\n--- {dataset.upper()} ---")
        suda_gmeans = [r["gmean"] for r in table1_raw[dataset]["SUDA"]]
        output.append(f"  SUDA G-mean: {np.mean(suda_gmeans):.4f} ± {np.std(suda_gmeans):.4f}")

        for baseline in baselines:
            if baseline not in table1_raw[dataset]:
                continue
            bl_gmeans = [r["gmean"] for r in table1_raw[dataset][baseline]]

            p_val, w_stat = wilcoxon_test(suda_gmeans, bl_gmeans)
            d = cohens_d(suda_gmeans, bl_gmeans)
            improvement = (np.mean(suda_gmeans) - np.mean(bl_gmeans)) / np.mean(bl_gmeans) * 100 if np.mean(bl_gmeans) > 0 else float('inf')

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

            output.append(f"  vs {baseline:20s}: Δ={np.mean(suda_gmeans)-np.mean(bl_gmeans):+.4f} ({improvement:+.1f}%), "
                         f"p={p_val:.4f} {sig}, d={d:.2f} ({effect_size_label(d)})")

    # ============================================================
    # TABLE 1: SUDA vs ARF - Attack Recall
    # ============================================================
    output.append("\n" + "=" * 70)
    output.append("TABLE 1: SUDA vs ARF - Attack Recall Comparison")
    output.append("=" * 70)

    for dataset in datasets:
        suda_ar = [r["attack_recall"] for r in table1_raw[dataset]["SUDA"]]
        arf_ar = [r["attack_recall"] for r in table1_raw[dataset]["ARF"]]

        p_val, _ = wilcoxon_test(suda_ar, arf_ar)
        d = cohens_d(suda_ar, arf_ar)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        output.append(f"  {dataset:12s}: SUDA {np.mean(suda_ar):.4f} vs ARF {np.mean(arf_ar):.4f}, "
                     f"Δ={np.mean(suda_ar)-np.mean(arf_ar):+.4f}, p={p_val:.4f} {sig}, d={d:.2f} ({effect_size_label(d)})")

    # ============================================================
    # TABLE 3: SUDA vs SUDA-NoUnlearn (Recovery)
    # ============================================================
    output.append("\n" + "=" * 70)
    output.append("TABLE 3: Unlearning Effect - Phase 3 Recovery")
    output.append("=" * 70)

    suda_p3 = [r["phase3_gmean"] for r in table3_raw["SUDA"]]
    nounlearn_p3 = [r["phase3_gmean"] for r in table3_raw["SUDA-NoUnlearn"]]
    arf_p3 = [r["phase3_gmean"] for r in table3_raw["ARF"]]
    srp_p3 = [r["phase3_gmean"] for r in table3_raw["SRP"]]

    output.append(f"  SUDA Phase 3:        {np.mean(suda_p3):.4f} ± {np.std(suda_p3):.4f}")
    output.append(f"  SUDA-NoUnlearn P3:   {np.mean(nounlearn_p3):.4f} ± {np.std(nounlearn_p3):.4f}")
    output.append(f"  ARF Phase 3:         {np.mean(arf_p3):.4f} ± {np.std(arf_p3):.4f}")
    output.append(f"  SRP Phase 3:         {np.mean(srp_p3):.4f} ± {np.std(srp_p3):.4f}")

    # SUDA vs SUDA-NoUnlearn
    p_val, _ = wilcoxon_test(suda_p3, nounlearn_p3)
    d = cohens_d(suda_p3, nounlearn_p3)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    output.append(f"\n  SUDA vs SUDA-NoUnlearn: Δ={np.mean(suda_p3)-np.mean(nounlearn_p3):+.4f}, "
                 f"p={p_val:.4f} {sig}, d={d:.2f} ({effect_size_label(d)})")

    # SUDA vs ARF (Phase 3)
    p_val, _ = wilcoxon_test(suda_p3, arf_p3)
    d = cohens_d(suda_p3, arf_p3)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    output.append(f"  SUDA vs ARF:           Δ={np.mean(suda_p3)-np.mean(arf_p3):+.4f}, "
                 f"p={p_val:.4f} {sig}, d={d:.2f} ({effect_size_label(d)})")

    # SUDA vs SRP (Phase 3)
    p_val, _ = wilcoxon_test(suda_p3, srp_p3)
    d = cohens_d(suda_p3, srp_p3)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    output.append(f"  SUDA vs SRP:           Δ={np.mean(suda_p3)-np.mean(srp_p3):+.4f}, "
                 f"p={p_val:.4f} {sig}, d={d:.2f} ({effect_size_label(d)})")

    # ============================================================
    # TABLE 3: Overall G-mean
    # ============================================================
    output.append("\n" + "-" * 40)
    output.append("TABLE 3: Overall G-mean Comparison")
    output.append("-" * 40)

    suda_overall = [r["overall_gmean"] for r in table3_raw["SUDA"]]
    nounlearn_overall = [r["overall_gmean"] for r in table3_raw["SUDA-NoUnlearn"]]
    arf_overall = [r["overall_gmean"] for r in table3_raw["ARF"]]
    srp_overall = [r["overall_gmean"] for r in table3_raw["SRP"]]

    p_val, _ = wilcoxon_test(suda_overall, nounlearn_overall)
    d = cohens_d(suda_overall, nounlearn_overall)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    output.append(f"  SUDA vs SUDA-NoUnlearn: Δ={np.mean(suda_overall)-np.mean(nounlearn_overall):+.4f}, "
                 f"p={p_val:.4f} {sig}, d={d:.2f} ({effect_size_label(d)})")

    p_val, _ = wilcoxon_test(suda_overall, arf_overall)
    d = cohens_d(suda_overall, arf_overall)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    output.append(f"  SUDA vs ARF:            Δ={np.mean(suda_overall)-np.mean(arf_overall):+.4f}, "
                 f"p={p_val:.4f} {sig}, d={d:.2f} ({effect_size_label(d)})")

    # ============================================================
    # CROSS-DATASET SUMMARY
    # ============================================================
    output.append("\n" + "=" * 70)
    output.append("CROSS-DATASET SUMMARY: SUDA vs ARF (G-mean)")
    output.append("=" * 70)

    all_suda = []
    all_arf = []
    for dataset in datasets:
        suda_g = [r["gmean"] for r in table1_raw[dataset]["SUDA"]]
        arf_g = [r["gmean"] for r in table1_raw[dataset]["ARF"]]
        all_suda.extend(suda_g)
        all_arf.extend(arf_g)

    stat_mw, p_val_mw = stats.mannwhitneyu(all_suda, all_arf, alternative='greater')
    d = cohens_d(all_suda, all_arf)
    sig = "***" if p_val_mw < 0.001 else "**" if p_val_mw < 0.01 else "*" if p_val_mw < 0.05 else "n.s."
    output.append(f"  Combined (15 runs): SUDA {np.mean(all_suda):.4f} vs ARF {np.mean(all_arf):.4f}")
    output.append(f"  Mann-Whitney U (one-sided): U={stat_mw:.0f}, p={p_val_mw:.6f} {sig}")
    output.append(f"  Cohen's d: {d:.2f} ({effect_size_label(d)})")

    # Permutation test for cross-dataset
    np.random.seed(42)
    p_perm = permutation_test(all_suda, all_arf)
    sig_perm = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else "n.s."
    output.append(f"  Permutation test (one-sided): p={p_perm:.4f} {sig_perm}")

    # ============================================================
    # SPEED COMPARISON
    # ============================================================
    output.append("\n" + "=" * 70)
    output.append("SPEED COMPARISON: SUDA vs ARF")
    output.append("=" * 70)

    for dataset in datasets:
        suda_t = [r["time_ms"] for r in table1_raw[dataset]["SUDA"]]
        arf_t = [r["time_ms"] for r in table1_raw[dataset]["ARF"]]
        speedup = np.mean(arf_t) / np.mean(suda_t)
        output.append(f"  {dataset:12s}: SUDA {np.mean(suda_t)/1000:.2f}s vs ARF {np.mean(arf_t)/1000:.1f}s = {speedup:.0f}x speedup")

    # Print all
    result_text = "\n".join(output)
    print(result_text)

    # Save to file
    with open(results_dir / "statistical_tests.txt", "w") as f:
        f.write(result_text)

    # Also save as JSON for paper
    stat_results = {
        "table1_suda_vs_arf": {},
        "table1_suda_vs_srp": {},
        "table3_recovery": {},
    }

    for dataset in datasets:
        suda_g = [r["gmean"] for r in table1_raw[dataset]["SUDA"]]
        arf_g = [r["gmean"] for r in table1_raw[dataset]["ARF"]]
        srp_g = [r["gmean"] for r in table1_raw[dataset]["SRP"]]

        p_arf, _ = wilcoxon_test(suda_g, arf_g)
        p_srp, _ = wilcoxon_test(suda_g, srp_g)
        d_arf = cohens_d(suda_g, arf_g)
        d_srp = cohens_d(suda_g, srp_g)

        stat_results["table1_suda_vs_arf"][dataset] = {
            "p_value": p_arf,
            "cohens_d": d_arf,
            "effect_size": effect_size_label(d_arf),
            "suda_mean": np.mean(suda_g),
            "arf_mean": np.mean(arf_g),
        }
        stat_results["table1_suda_vs_srp"][dataset] = {
            "p_value": p_srp,
            "cohens_d": d_srp,
            "effect_size": effect_size_label(d_srp),
            "suda_mean": np.mean(suda_g),
            "srp_mean": np.mean(srp_g),
        }

    p_recovery, _ = wilcoxon_test(suda_p3, nounlearn_p3)
    d_recovery = cohens_d(suda_p3, nounlearn_p3)
    stat_results["table3_recovery"] = {
        "suda_vs_nounlearn": {
            "p_value": p_recovery,
            "cohens_d": d_recovery,
            "effect_size": effect_size_label(d_recovery),
        }
    }

    with open(results_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'statistical_tests.txt'}")
    print(f"JSON saved to {results_dir / 'statistical_tests.json'}")


if __name__ == "__main__":
    main()
