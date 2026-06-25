"""
aggregate_results.py
====================
Agregasi hasil eksperimen RL multi-seed untuk laporan TA.

Cara pakai:
    python experiments/aggregate_results.py
    python experiments/aggregate_results.py --last-n 10 --out results/summary.csv

Output:
    - Tabel mean±std per (scenario, agent) dari N episode terakhir tiap seed
    - Uji statistik Wilcoxon signed-rank: GAT vs IDQN, GAT vs PKJI, IDQN vs PKJI
    - CSV ringkasan + print ke terminal

Kolom output:
    scenario | agent | n_seeds | mean_reward | std_reward |
    mean_delay | std_delay | mean_queue | std_queue |
    mean_throughput | std_throughput | mean_teleport | std_teleport |
    p_gat_vs_idqn | p_gat_vs_pkji | p_idqn_vs_pkji
"""

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

# scipy opsional — fallback ke pesan jika tidak ada
try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy tidak terinstall — uji statistik dilewati. Install: pip install scipy")

LOGS_DIR = Path("d:/ITB/TA/gnn-marl-traffic/logs")

# ---------------------------------------------------------------------------
# Registry run final yang valid per (scenario, agent, seed)
# Tambahkan entry baru di sini setelah run selesai.
# Format: (scenario, agent, seed) -> log_dir_name
# ---------------------------------------------------------------------------
FINAL_RUNS = {
    # === arterial_stable ===
    ("arterial_stable", "gat_dqn",        42):  "gat_dqn_arterial_stable_20260527_094734_arterial_gat_fix50_fresh_clip1_decay096_s42",
    ("arterial_stable", "gat_dqn",       123):  "gat_dqn_arterial_stable_20260527_094734_arterial_gat_fix50_fresh_clip1_decay096_s123",
    ("arterial_stable", "independent_dqn", 42):  "independent_dqn_arterial_stable_20260528_035921_arterial_idqn_final_decay098_s42",
    ("arterial_stable", "independent_dqn",123):  "independent_dqn_arterial_stable_20260528_035921_arterial_idqn_final_decay098_s123",

    # === arterial_peak ===
    ("arterial_peak",   "gat_dqn",        42):  "gat_dqn_arterial_peak_20260529_070833_arterial_peak_gat_s42",
    ("arterial_peak",   "gat_dqn",       123):  "gat_dqn_arterial_peak_20260529_070833_arterial_peak_gat_s123",
    ("arterial_peak",   "independent_dqn", 42):  "independent_dqn_arterial_peak_20260529_070833_arterial_peak_idqn_s42",
    ("arterial_peak",   "independent_dqn",123):  "independent_dqn_arterial_peak_20260529_070833_arterial_peak_idqn_s123",

    # === arterial_unbalanced (diisi setelah run selesai) ===
    # ("arterial_unbalanced", "gat_dqn",        42):  "...",
    # ("arterial_unbalanced", "gat_dqn",       123):  "...",
    # ("arterial_unbalanced", "gat_dqn",        77):  "...",
    # ("arterial_unbalanced", "gat_dqn",       111):  "...",
    # ("arterial_unbalanced", "independent_dqn", 42):  "...",
    # ("arterial_unbalanced", "independent_dqn",123):  "...",
    # ("arterial_unbalanced", "independent_dqn", 77):  "...",
    # ("arterial_unbalanced", "independent_dqn",111):  "...",
}

# PKJI fixed-time baseline — pakai run terpanjang (5 episode, semua dipakai)
PKJI_BASELINES = {
    "arterial_stable": "fixed_time_arterial_stable_20260528_171443_pkji_arterial_baseline_long",
    # "arterial_peak":   "",        # isi jika ada run PKJI peak
    # "arterial_unbalanced": "",    # isi jika ada run PKJI unbalanced
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_metrics(log_dir: Path) -> list[dict]:
    """Baca metrics.csv, return list of row dicts."""
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv tidak ada: {csv_path}")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def last_n_rows(rows: list[dict], n: int) -> list[dict]:
    """Ambil N episode terakhir (sudah konvergen)."""
    return rows[-n:] if len(rows) >= n else rows


def episode_stats(rows: list[dict]) -> dict:
    """Hitung mean metrik dari list rows (satu run, N episode)."""
    rewards     = [safe_float(r.get("reward")) for r in rows]
    delays      = [safe_float(r.get("avg_delay")) for r in rows]
    queues      = [safe_float(r.get("avg_queue")) for r in rows]
    throughputs = [safe_float(r.get("throughput")) for r in rows]
    teleports   = [safe_float(r.get("teleport_started", 0)) for r in rows]
    return {
        "mean_reward":     statistics.mean(rewards),
        "mean_delay":      statistics.mean(delays),
        "mean_queue":      statistics.mean(queues),
        "mean_throughput": statistics.mean(throughputs),
        "total_teleport":  sum(teleports),
        # simpan raw untuk wilcoxon
        "_rewards":     rewards,
        "_delays":      delays,
        "_queues":      queues,
        "_throughputs": throughputs,
    }


def multi_seed_stats(per_seed: list[dict], key: str):
    """Mean dan std dari metrik `key` across seeds."""
    vals = [s[key] for s in per_seed]
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def wilcoxon_p(a_vals: list[float], b_vals: list[float]) -> float | str:
    """
    Wilcoxon signed-rank test antara dua list nilai.
    a_vals, b_vals: list episode-level values (digabung across seeds).
    Return p-value atau 'N/A'.
    """
    if not HAS_SCIPY:
        return "N/A (no scipy)"
    if len(a_vals) != len(b_vals):
        # pad ke panjang minimum
        n = min(len(a_vals), len(b_vals))
        a_vals, b_vals = a_vals[:n], b_vals[:n]
    if len(a_vals) < 2:
        return "N/A (n<2)"
    try:
        stat, p = wilcoxon(a_vals, b_vals, alternative="two-sided")
        return round(p, 4)
    except Exception as e:
        return f"ERR:{e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate(last_n: int, out_path: str | None):
    scenarios = sorted({s for s, _, _ in FINAL_RUNS})
    agents    = ["gat_dqn", "independent_dqn"]

    summary_rows = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*60}")

        # --- PKJI baseline ---
        pkji_stats = None
        if scenario in PKJI_BASELINES:
            pkji_dir = LOGS_DIR / PKJI_BASELINES[scenario]
            try:
                pkji_rows = read_metrics(pkji_dir)
                pkji_stats = episode_stats(pkji_rows)   # pakai semua episode
                r, d, q, t = (
                    pkji_stats["mean_reward"],
                    pkji_stats["mean_delay"],
                    pkji_stats["mean_queue"],
                    pkji_stats["mean_throughput"],
                )
                print(f"\n  [PKJI baseline] n_ep={len(pkji_rows)}")
                print(f"    reward={r:.2f}  delay={d:.2f}s  queue={q:.2f}  throughput={t:.0f}")
                summary_rows.append({
                    "scenario": scenario,
                    "agent": "pkji_fixed_time",
                    "n_seeds": 1,
                    "n_episodes_per_seed": len(pkji_rows),
                    "mean_reward": round(r, 4),
                    "std_reward": 0.0,
                    "mean_delay": round(d, 4),
                    "std_delay": 0.0,
                    "mean_queue": round(q, 4),
                    "std_queue": 0.0,
                    "mean_throughput": round(t, 1),
                    "std_throughput": 0.0,
                    "mean_teleport": round(pkji_stats["total_teleport"], 2),
                    "std_teleport": 0.0,
                    "p_vs_gat": "ref",
                    "p_vs_idqn": "ref",
                })
            except FileNotFoundError as e:
                print(f"  [PKJI] SKIP: {e}")

        # --- RL agents ---
        agent_data = {}  # agent -> {"per_seed": [...], "all_rewards": [...], ...}

        for agent in agents:
            seeds_found = [(s, k) for (sc, ag, s), k in FINAL_RUNS.items()
                          if sc == scenario and ag == agent]
            if not seeds_found:
                print(f"\n  [{agent}] tidak ada run terdaftar — skip")
                continue

            per_seed_stats = []
            all_rewards, all_delays, all_queues = [], [], []

            for seed, log_name in sorted(seeds_found):
                log_dir = LOGS_DIR / log_name
                try:
                    rows = read_metrics(log_dir)
                    tail = last_n_rows(rows, last_n)
                    s    = episode_stats(tail)
                    per_seed_stats.append(s)
                    all_rewards.extend(s["_rewards"])
                    all_delays.extend(s["_delays"])
                    all_queues.extend(s["_queues"])
                    print(f"  [{agent}] seed={seed:4d} | ep_total={len(rows):3d} | "
                          f"last{last_n}: reward={s['mean_reward']:7.2f}  "
                          f"delay={s['mean_delay']:5.2f}s  "
                          f"queue={s['mean_queue']:4.2f}  "
                          f"teleport={s['total_teleport']:.0f}")
                except FileNotFoundError as e:
                    print(f"  [{agent}] seed={seed} SKIP: {e}")

            if not per_seed_stats:
                continue

            agent_data[agent] = {
                "per_seed": per_seed_stats,
                "all_rewards": all_rewards,
                "all_delays":  all_delays,
                "all_queues":  all_queues,
            }

            mr, sr = multi_seed_stats(per_seed_stats, "mean_reward")
            md, sd = multi_seed_stats(per_seed_stats, "mean_delay")
            mq, sq = multi_seed_stats(per_seed_stats, "mean_queue")
            mt, st = multi_seed_stats(per_seed_stats, "mean_throughput")
            mtel   = statistics.mean([s["total_teleport"] for s in per_seed_stats])
            stel   = (statistics.stdev([s["total_teleport"] for s in per_seed_stats])
                      if len(per_seed_stats) > 1 else 0.0)

            print(f"  [{agent}] AGREGAT n_seeds={len(per_seed_stats)}")
            print(f"    reward={mr:.2f}±{sr:.2f}  delay={md:.2f}±{sd:.2f}s  "
                  f"queue={mq:.2f}±{sq:.2f}  throughput={mt:.0f}±{st:.0f}  "
                  f"teleport={mtel:.1f}±{stel:.1f}")

            summary_rows.append({
                "scenario": scenario,
                "agent": agent,
                "n_seeds": len(per_seed_stats),
                "n_episodes_per_seed": last_n,
                "mean_reward": round(mr, 4),
                "std_reward":  round(sr, 4),
                "mean_delay":  round(md, 4),
                "std_delay":   round(sd, 4),
                "mean_queue":  round(mq, 4),
                "std_queue":   round(sq, 4),
                "mean_throughput": round(mt, 1),
                "std_throughput":  round(st, 1),
                "mean_teleport":   round(mtel, 2),
                "std_teleport":    round(stel, 2),
                "p_vs_gat":   "ref" if agent == "gat_dqn" else None,
                "p_vs_idqn":  "ref" if agent == "independent_dqn" else None,
            })

        # --- Uji statistik antar agent ---
        print(f"\n  [STATISTIK] scenario={scenario}")
        gat  = agent_data.get("gat_dqn")
        idqn = agent_data.get("independent_dqn")

        if gat and idqn:
            p_reward = wilcoxon_p(gat["all_rewards"], idqn["all_rewards"])
            p_delay  = wilcoxon_p(gat["all_delays"],  idqn["all_delays"])
            p_queue  = wilcoxon_p(gat["all_queues"],  idqn["all_queues"])
            print(f"    GAT vs IDQN  — reward p={p_reward}  delay p={p_delay}  queue p={p_queue}")
            # update p ke summary rows
            for row in summary_rows:
                if row["scenario"] == scenario and row["agent"] == "independent_dqn":
                    row["p_vs_gat"] = p_reward
            for row in summary_rows:
                if row["scenario"] == scenario and row["agent"] == "gat_dqn":
                    row["p_vs_idqn"] = p_reward

        if gat and pkji_stats:
            pkji_rewards = pkji_stats["_rewards"] * max(1, len(gat["all_rewards"]) // len(pkji_stats["_rewards"]))
            p_pkji = wilcoxon_p(gat["all_rewards"], pkji_rewards[:len(gat["all_rewards"])])
            print(f"    GAT vs PKJI  — reward p={p_pkji}")

        if idqn and pkji_stats:
            pkji_rewards = pkji_stats["_rewards"] * max(1, len(idqn["all_rewards"]) // len(pkji_stats["_rewards"]))
            p_pkji = wilcoxon_p(idqn["all_rewards"], pkji_rewards[:len(idqn["all_rewards"])])
            print(f"    IDQN vs PKJI — reward p={p_pkji}")

    # --- Tulis CSV ---
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "scenario", "agent", "n_seeds", "n_episodes_per_seed",
            "mean_reward", "std_reward",
            "mean_delay",  "std_delay",
            "mean_queue",  "std_queue",
            "mean_throughput", "std_throughput",
            "mean_teleport",   "std_teleport",
            "p_vs_gat", "p_vs_idqn",
        ]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[OK] CSV tersimpan: {out}")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agregasi hasil eksperimen RL multi-seed")
    parser.add_argument(
        "--last-n", type=int, default=10,
        help="Jumlah episode terakhir per seed untuk agregasi (default: 10)",
    )
    parser.add_argument(
        "--out", type=str, default="results/summary.csv",
        help="Path output CSV (default: results/summary.csv)",
    )
    args = parser.parse_args()
    aggregate(last_n=args.last_n, out_path=args.out)
