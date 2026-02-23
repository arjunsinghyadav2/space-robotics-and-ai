import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy import linalg

def _read_defaults():
    #Bryson-style tuning baseline for R and disturbance settings.
    r = 0.1
    amp = 15.0
    fr = np.array([0.5, 4.0], dtype=float)
    return r, amp, fr

def _gain(a, b, q, r):
    p = linalg.solve_continuous_are(a, b, np.diag(q), np.array([[r]], dtype=float))
    return (np.array([[1.0 / r]]) @ b.T @ p).astype(np.float32).reshape(-1)

def _disturbance(steps, dt, amp, f_lo, f_hi, seed):
    #multi-sine + random amplitude + gaussian noise
    rng = np.random.default_rng(seed)
    t = np.arange(steps) * dt
    freqs = rng.uniform(f_lo, f_hi, 5)
    phases = rng.uniform(0.0, 2.0 * np.pi, 5)
    d = np.zeros(steps, dtype=np.float32)
    for f, p in zip(freqs, phases):
        d += (amp * rng.uniform(0.8, 1.2, size=steps) * np.sin(2.0 * np.pi * f * t + p)).astype(np.float32)
    d += rng.normal(0.0, amp * 0.1, size=steps).astype(np.float32)
    return d

def _rollout_batch(k_batch, d, dt, seconds, device, u_limit=400.0):
    #linearized dynamics as lqr_controller A,B for M=m=L=1
    a = torch.tensor([[0, 1, 0, 0], [0, 0, 9.81, 0], [0, 0, 0, 1], [0, 0, 19.62, 0]], dtype=torch.float32, device=device)
    b = torch.tensor([0, 1, 0, -1], dtype=torch.float32, device=device)
    steps = int(seconds / dt)
    n = k_batch.shape[0]
    x = torch.zeros((n, 4), dtype=torch.float32, device=device)
    x[:, 2] = 0.03
    alive = torch.ones(n, dtype=torch.bool, device=device)
    surv = torch.full((n,), seconds, dtype=torch.float32, device=device)
    max_x = torch.zeros(n, dtype=torch.float32, device=device)
    max_th = torch.zeros(n, dtype=torch.float32, device=device)
    u_sum = torch.zeros(n, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)
    th_lim = torch.tensor(np.deg2rad(45.0), dtype=torch.float32, device=device)

    for i in range(steps):
        u = -(x * k_batch).sum(dim=1)
        u = torch.clamp(u, -u_limit, u_limit)
        x = x + dt * (x @ a.T + (u + d[i]).unsqueeze(1) * b)
        max_x = torch.maximum(max_x, x[:, 0].abs())
        max_th = torch.maximum(max_th, x[:, 2].abs())
        u_sum += u.abs()
        crossed = alive & ((x[:, 0].abs() > 2.5) | (x[:, 2].abs() > th_lim))
        surv = torch.where(crossed, torch.full_like(surv, i * dt), surv)
        alive = alive & (~crossed)

    avg_u = u_sum / steps
    max_th_deg = max_th * (180.0 / np.pi)
    score = surv - 0.5 * max_x - 0.02 * max_th_deg - 0.03 * avg_u
    return surv.cpu().numpy(), max_x.cpu().numpy(), max_th_deg.cpu().numpy(), avg_u.cpu().numpy(), score.cpu().numpy()

def main():
    p = argparse.ArgumentParser(description="LQR tuner")
    p.add_argument("--seconds", type=float, default=30.0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r0, amp, fr = _read_defaults()
    #Bryson tolerance sweep
    #candidate count: 8 x 6 x 6 x 6 x 7 = 12096
    x_tols = np.array([0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.2, 2.5], dtype=float)
    xdot_tols = np.array([1.0, 1.5, 2.0, 2.5, 3.5, 5.0], dtype=float)
    theta_tols = np.array([0.20, 0.25, 0.30, 0.35, 0.42, 0.50], dtype=float)  # rad
    thetadot_tols = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8], dtype=float)  # rad/s
    r_scales = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8], dtype=float)
    A = np.array([[0, 1, 0, 0], [0, 0, 9.81, 0], [0, 0, 0, 1], [0, 0, 19.62, 0]], dtype=float)
    B = np.array([[0], [1], [0], [-1]], dtype=float)

    cand = []
    for x_tol in x_tols:
        for xdot_tol in xdot_tols:
            for theta_tol in theta_tols:
                for thetadot_tol in thetadot_tols:
                    q = np.array(
                        [
                            1.0 / (x_tol ** 2),
                            1.0 / (xdot_tol ** 2),
                            1.0 / (theta_tol ** 2),
                            1.0 / (thetadot_tol ** 2),
                        ],
                        dtype=float,
                    )
                    for rs in r_scales:
                        cand.append((q, r0 * rs, float(x_tol), float(xdot_tol), float(theta_tol), float(thetadot_tol), float(rs)))
    ks = np.stack([_gain(A, B, q, r) for q, r, _, _, _, _, _ in cand])
    k_batch = torch.tensor(ks, device=device)
    seeds = [0, 1, 2]
    worst_surv = np.full(len(cand), args.seconds, dtype=np.float32)
    worst_mx = np.zeros(len(cand), dtype=np.float32)
    worst_mth = np.zeros(len(cand), dtype=np.float32)
    worst_uavg = np.zeros(len(cand), dtype=np.float32)
    for seed in seeds:
        d = _disturbance(int(args.seconds / 0.01), 0.01, amp, float(fr[0]), float(fr[1]), seed=seed)
        surv, mx, mth, uavg, _ = _rollout_batch(k_batch, d, dt=0.01, seconds=args.seconds, device=device)
        worst_surv = np.minimum(worst_surv, surv)
        worst_mx = np.maximum(worst_mx, mx)
        worst_mth = np.maximum(worst_mth, mth)
        worst_uavg = np.maximum(worst_uavg, uavg)

    score = worst_surv - 0.5 * worst_mx - 0.02 * worst_mth - 0.03 * worst_uavg
    success = worst_surv >= (args.seconds - 0.01)
    best_i = int(np.argmax(np.where(success, score, -1e9))) if success.any() else int(np.argmax(score))

    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (q, r, x_tol, xdot_tol, theta_tol, thetadot_tol, rs) in enumerate(cand):
        rows.append(
            {"id": i,
                "x_tol_m": x_tol,
                "x_dot_tol_mps": xdot_tol,
                "theta_tol_rad": theta_tol,
                "theta_dot_tol_rps": thetadot_tol,
                "q_theta_over_x": float(q[2] / q[0]),
                "q_theta_dot_over_x_dot": float(q[3] / q[1]),
                "r_scale": rs,
                "q": q.tolist(),
                "r": float(r),
                "survival_s": float(worst_surv[i]),
                "max_cart_m": float(worst_mx[i]),
                "max_theta_deg": float(worst_mth[i]),
                "avg_abs_u": float(worst_uavg[i]),
                "score": float(score[i]),
                "success": bool(success[i]),})

    with (out / "tuning_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best = rows[best_i]
    with (out / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump({"device": device, "best": best}, f, indent=2)

    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
