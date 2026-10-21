#!/usr/bin/env python3
"""
Disciplined 13-trit clock: holdover simulation.

A 13-trit counter wraps once per 12 hours, so it ticks at exactly
3^13 / 43200 = 59049/1600 Hz (about 36.9 Hz). It is driven by a simulated
crystal whose true frequency depends on offset, aging, temperature and a
random walk. For the first few days a reference (think GPS 1PPS) is
available. A Kalman filter keeps a model of the oscillator:

    state = [phase error (ticks), freq offset (ppm),
             aging drift (ppm/day), temp coefficient (ppm/degC)]

Each measurement's residual (measured - predicted phase) updates the model.
Then the reference is removed (holdover) and we compare:

    free-run : the raw counter, never corrected
    basic    : phase + frequency only (the "learn the offset" clock)
    full     : phase + frequency + aging + temperature model

Run:  python3 ternary_clock_sim.py
Needs numpy; matplotlib is optional (saves ternary_clock_holdover.png).
"""
import numpy as np

TRITS = 13
MOD = 3 ** TRITS                  # 1,594,323 states
DAY = 86400.0
RATE = MOD / 43200.0              # ticks per second (= 59049/1600)
T0 = 25.0                         # reference temperature, deg C
RESID_TRITS = 6                   # balanced-ternary residual register
RESID_MAX = (3 ** RESID_TRITS - 1) // 2   # +/-364 ticks


# ---------------------------------------------------------------- trit helpers
def to_ternary(n, width=TRITS):
    """Unsigned ternary digits, most significant first (the clock 'face')."""
    n %= 3 ** width
    d = []
    for _ in range(width):
        d.append(str(n % 3))
        n //= 3
    return "".join(reversed(d))


def to_balanced(n, width=RESID_TRITS):
    """Balanced ternary with symbols - 0 +  (digits -1, 0, +1), MSB first."""
    d = []
    for _ in range(width):
        r = n % 3
        if r == 2:
            d.append("-")
            n = (n + 1) // 3
        else:
            d.append("0" if r == 0 else "+")
            n //= 3
    return "".join(reversed(d))


def from_balanced(s):
    v = 0
    for ch in s:
        v = v * 3 + {"-": -1, "0": 0, "+": 1}[ch]
    return v


for _n in range(-RESID_MAX, RESID_MAX + 1):
    assert from_balanced(to_balanced(_n)) == _n   # self-test: round-trips


# ------------------------------------------------------------ the discipline
class Discipline:
    """Kalman filter over [phase(ticks), f(ppm), drift(ppm/day), tempco(ppm/C)].

    With full_model=False the drift and tempco states are pinned at zero,
    which gives the plain phase+frequency clock for comparison.
    """

    def __init__(self, full_model, dt):
        self.dt = dt
        self.x = np.zeros(4)
        self.P = np.diag([50.0 ** 2, 10.0 ** 2,
                          0.5 ** 2 if full_model else 0.0,
                          1.0 ** 2 if full_model else 0.0])
        s = np.sqrt(dt / DAY)
        self.Q = np.diag([0.02 ** 2, (0.015 * s) ** 2,
                          (0.002 * s) ** 2 if full_model else 0.0,
                          1e-14 if full_model else 0.0])
        self.R = 0.13             # quantization (~0.29^2) + jitter (0.2^2)

    def predict(self, temp_c):
        dt, tc = self.dt, temp_c - T0
        k = RATE * dt * 1e-6
        F = np.eye(4)
        F[0, 1] = k
        F[0, 2] = 0.5 * k * dt / DAY
        F[0, 3] = k * tc
        F[1, 2] = dt / DAY
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """z = measured phase difference (ticks). Returns the residual."""
        resid = z - self.x[0]
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S
        self.x = self.x + K * resid
        self.P = self.P - np.outer(K, self.P[0, :])
        return resid


# ----------------------------------------------------------------- simulation
def run(seed, days_ref=3.0, days_hold=2.0, dt=60.0):
    rng = np.random.default_rng(seed)
    n = int((days_ref + days_hold) * DAY / dt)
    t_loss = days_ref * DAY

    # hidden truth the filter has to discover
    y0 = rng.uniform(1.0, 3.0)                         # ppm offset
    aging = rng.uniform(0.02, 0.08)                    # ppm/day
    k_true = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])   # ppm/degC
    rw_step = 0.01 * np.sqrt(dt / DAY)                 # ppm, random-walk FM noise

    # temperature: diurnal swing + slow weather, sensor noise on top
    a = np.exp(-dt / 21600.0)
    slow = np.zeros(n)
    for i in range(1, n):
        slow[i] = a * slow[i - 1] + 0.7 * np.sqrt(1 - a * a) * rng.normal()
    t = np.arange(n) * dt
    temp_true = T0 + 3.0 * np.sin(2 * np.pi * t / DAY + rng.uniform(0, 6.28)) + slow
    temp_meas = temp_true + rng.normal(0, 0.05, n)

    basic, full = Discipline(False, dt), Discipline(True, dt)
    local = rng.uniform(-5, 5)                         # starting phase error, ticks
    y_rw = 0.0
    tt = np.zeros(n)
    err_free, err_basic, err_full = np.zeros(n), np.zeros(n), np.zeros(n)
    n_sat = n_resid = 0

    for i in range(n):
        y_rw += rw_step * rng.normal()
        y = y0 + aging * (t[i] / DAY) + k_true * (temp_true[i] - T0) + y_rw
        local += RATE * dt * (1 + y * 1e-6)            # true counter phase
        tt[i] = (i + 1) * dt
        ref = RATE * tt[i]                             # ideal phase at this instant

        basic.predict(temp_meas[i])
        full.predict(temp_meas[i])
        if tt[i] <= t_loss:                            # reference available
            z = np.round(local - ref + rng.normal(0, 0.2))   # integer counter reading
            basic.update(z)
            r = full.update(z)
            n_resid += 1
            if abs(int(np.round(r))) > RESID_MAX:
                n_sat += 1                             # would overflow 6-trit register

        err_free[i] = local - ref
        err_basic[i] = local - ref - basic.x[0]
        err_full[i] = local - ref - full.x[0]

    return dict(tt=tt, t_loss=t_loss, days_hold=days_hold, free=err_free,
                basic=err_basic, full=err_full, filt=full, k_true=k_true,
                aging=aging, y0=y0, local=local, n_sat=n_sat, n_resid=n_resid)


def ms(ticks):
    return ticks / RATE * 1000.0


def hold_hours(res, key, thresh_ticks=1.0):
    """Hours of holdover before |error| exceeds thresh (capped at window)."""
    m = res["tt"] > res["t_loss"]
    e, t = np.abs(res[key][m]), res["tt"][m]
    bad = np.nonzero(e > thresh_ticks)[0]
    cap = res["days_hold"] * 24
    return cap if len(bad) == 0 else (t[bad[0]] - res["t_loss"]) / 3600.0


def main():
    print(f"Counter: {TRITS} trits = {MOD:,} states, wraps every 12 h")
    print(f"Tick rate: 59049/1600 Hz = {RATE:.6f} Hz  (1 tick = {1000/RATE:.2f} ms)\n")

    r = run(seed=0)
    f = r["filt"]
    m = r["tt"] > r["t_loss"]
    last_day = (r["tt"] > r["t_loss"] - DAY) & (r["tt"] <= r["t_loss"])
    print("=== Seed 0 detail ===")
    print(f"learned tempco : {f.x[3]:+.3f} ppm/C   (true {r['k_true']:+.3f})")
    print(f"learned aging  : {f.x[2]:+.3f} ppm/day (true {r['aging']:+.3f})")
    print(f"RMS error, last training day (full model): "
          f"{ms(np.sqrt(np.mean(r['full'][last_day] ** 2))):.1f} ms")
    print(f"residual register ({RESID_TRITS} trits, +/-{RESID_MAX}): "
          f"{r['n_sat']} saturations in {r['n_resid']} updates")
    print(f"\nHoldover ({r['days_hold']:.0f} days, reference removed), end-of-window error:")
    for key, label in [("free", "free-run"), ("basic", "phase+freq"), ("full", "full model")]:
        print(f"  {label:<11}: {ms(r[key][-1]):+9.1f} ms   "
              f"({r[key][-1]:+.2f} ticks)   "
              f"time to >1 tick: {hold_hours(r, key):5.1f} h")

    # the clock face at the end, as the corrected counter would show it
    face = int(np.floor(r["local"] - f.x[0])) % MOD
    secs = face / RATE
    hh, rem = divmod(int(secs), 3600)
    print(f"\nFace at end of run: {to_ternary(face)} (ternary) "
          f"= {hh % 12:02d}:{rem // 60:02d}:{rem % 60:02d} on a 12 h dial")
    print(f"Last residual in balanced ternary: "
          f"{to_balanced(int(np.clip(np.round(r['full'][-1]), -RESID_MAX, RESID_MAX)))}")

    # many seeds, so one lucky run doesn't mislead
    seeds = range(1, 16)
    runs = [run(s) for s in seeds]
    print(f"\n=== {len(runs)} random oscillators (different offset/aging/tempco) ===")
    print(f"{'':<11}{'median final err':>18}{'median time to >1 tick':>26}")
    for key, label in [("free", "free-run"), ("basic", "phase+freq"), ("full", "full model")]:
        fe = np.median([abs(ms(x[key][-1])) for x in runs])
        ht = np.median([hold_hours(x, key) for x in runs])
        print(f"{label:<11}{fe:>15.1f} ms{ht:>23.1f} h")
    print("\n(time to >1 tick is capped at the 48 h holdover window)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        th = r["tt"] / 3600
        for key, label in [("free", "free-run"), ("basic", "phase+freq"), ("full", "full model")]:
            ax.plot(th, ms(r[key]), label=label, lw=1)
        ax.axvline(r["t_loss"] / 3600, color="k", ls="--", lw=0.8)
        ax.text(r["t_loss"] / 3600 + 1, 0, "reference lost", rotation=90, va="bottom", fontsize=8)
        ax.axhspan(-1000 / RATE, 1000 / RATE, color="gray", alpha=0.15, label="+/-1 tick")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("hours")
        ax.set_ylabel("time error (ms)")
        ax.set_title("Disciplined 13-trit clock: holdover")
        ax.legend()
        fig.tight_layout()
        fig.savefig("ternary_clock_holdover.png", dpi=130)
        print("\nSaved ternary_clock_holdover.png")
    except ImportError:
        print("\n(matplotlib not installed; skipped plot)")


if __name__ == "__main__":
    main()
