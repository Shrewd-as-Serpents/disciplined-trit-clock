#!/usr/bin/env python3
"""
Disciplined 13-trit clock, v5: power-of-three scale variant, built against the exhaustively
verified accumulator (see ternary_accum_verify.py), with a four-entry information ledger.

v1 kept the Kalman state in floating point and only *checked* whether a ternary
residual would fit. Here the estimator's state and its innovation (the
"differential", measured minus predicted phase) are stored in balanced-ternary
registers of fixed width and fixed least-significant-digit (lsb) size:

    - every value is rounded to the register grid (nearest, i.e. balanced-ternary
      truncation) and CLIPPED at the register's range (saturation, never a wrap)
    - the correction applied to the state is K * (decoded innovation register)
    - the state carried from step to step is the decoded state registers

What stays in floating point: the covariance matrix P and the gain K. So this is
"ternary state and innovation path", not "all arithmetic in trits".

Compared:  free-run | phase+freq (float) | full (float, unrestricted)
           | full ternary (13-trit state, 6-trit innovation, saturating)
           | full ternary (13-trit state, 13-trit innovation = quantized, no clip)
           | full ternary (9-trit state, 6-trit innovation)

Scenarios deliberately make the world disagree with the estimator's model.

New in v3: a frequency register can carry a REMAINDER (guard digits) so aging increments
smaller than half an lsb accumulate instead of being rounded away every step. The remainder
is real storage (guard trits) and is counted; an uncounted float remainder is included only
as a diagnostic upper bound.

v4: the increment that feeds the frequency register is computed from the aging register INTEGER
with exact integer arithmetic (an exact rational constant c = cn/cd; units = nearest(q_age*c*D)).
No float touches that path; the widest intermediate is measured and reported. A diagnostic
counts how often the old float-derived path would have produced a different integer.

v5: scale3=k chooses the aging register scale so the aging->frequency coupling is exactly 3^-k,
making the increment a balanced-ternary digit-position change (no multiplier, no ties, exactly
sign-symmetric; exact when guard g >= k). shift=True uses the shift form that the verifier proved
equal to the generic integer formula and the Fraction reference; shift=False runs the generic
formula with the same scale, and the two must be bit-identical.

Run:  python3 ternary_clock_sim_v5.py [n_seeds]      (default 12)
"""
import os
import sys
import time
from math import gcd
from concurrent.futures import ProcessPoolExecutor

from fractions import Fraction

import numpy as np

TRITS = 13
MOD = 3 ** TRITS                  # 1,594,323
DAY = 86400.0
RATE = MOD / 43200.0              # ticks per second = 59049/1600
T0 = 25.0

# register ranges (what +/- full scale means for each state element)
PHASE_RANGE = 256.0               # ticks
FREQ_RANGE = 20.0                 # ppm
AGE_RANGE = 0.5                   # ppm/day
TC_RANGE = 2.0                    # ppm/degC
INN_LSB = 1.0 / 32.0              # innovation register resolution, ticks


# ---------------------------------------------------------------- trit helpers
def half(w):
    """Largest magnitude of a w-trit balanced-ternary register."""
    return (3 ** w - 1) // 2


def to_ternary(n, width=TRITS):
    n %= 3 ** width
    d = []
    for _ in range(width):
        d.append(str(n % 3))
        n //= 3
    return "".join(reversed(d))


def to_balanced(n, width):
    """Balanced ternary, symbols - 0 +, most significant first."""
    assert abs(n) <= half(width), "value does not fit register"
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


for _w in (4, 6, 9, 13):
    for _n in list(range(-min(half(_w), 400), min(half(_w), 400) + 1)) + [half(_w), -half(_w)]:
        assert from_balanced(to_balanced(_n, _w)) == _n    # round-trip self-test


def rate_pattern():
    """Periods of the exact fractional-phase pattern from 59049/1600 ticks/s."""
    num, den = 59049, 1600
    return den // gcd(num, den), den // gcd(num * 60, den)


# ------------------------------------------------------------ the estimators
class Discipline:
    """Kalman filter over [phase(ticks), f(ppm), drift(ppm/day), tempco(ppm/C)].
    Floating point throughout (the 'unrestricted' baseline)."""

    def __init__(self, full_model, dt, clip=None):
        self.dt = dt
        self.clip = clip                  # optional innovation clip (ticks), no quantization
        self.x = np.zeros(4)
        self.P = np.diag([50.0 ** 2, 10.0 ** 2,
                          0.5 ** 2 if full_model else 0.0,
                          1.0 ** 2 if full_model else 0.0])
        s = np.sqrt(dt / DAY)
        self.Q = np.diag([0.02 ** 2, (0.015 * s) ** 2,
                          (0.002 * s) ** 2 if full_model else 0.0,
                          1e-14 if full_model else 0.0])
        self.R = 0.13
        self.last = 0.0

    def _F(self, temp_c):
        dt, tc = self.dt, temp_c - T0
        k = RATE * dt * 1e-6
        F = np.eye(4)
        F[0, 1] = k
        F[0, 2] = 0.5 * k * dt / DAY
        F[0, 3] = k * tc
        F[1, 2] = dt / DAY
        return F

    def predict(self, temp_c):
        F = self._F(temp_c)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        resid = z - self.x[0]
        used = resid if self.clip is None else float(np.clip(resid, -self.clip, self.clip))
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S
        self.x = self.x + K * used
        self.P = self.P - np.outer(K, self.P[0, :])
        self.last = used
        return used


class TernaryDiscipline(Discipline):
    """Same filter, but state and innovation live in balanced-ternary registers.

    state registers : phase (13 trits), then freq/aging/tempco (w_state trits each)
    innovation reg  : w_inn trits, lsb = INN_LSB ticks, saturating
    """

    def __init__(self, full_model, dt, w_state=13, w_inn=6, w_phase=13, guard=None, w_freq=None,
                 scale3=None, shift=False):
        super().__init__(full_model, dt)
        widths = np.array([w_phase, w_freq or w_state, w_state, w_state])
        ranges = np.array([PHASE_RANGE, FREQ_RANGE, AGE_RANGE, TC_RANGE])
        self.widths = widths
        self.lim = np.array([half(int(w)) for w in widths])
        self.lsb = ranges / self.lim
        self.q = np.zeros(4, dtype=np.int64)       # the actual register contents
        self.w_inn = w_inn
        self.inn_lim = half(w_inn)
        self.inn_q = 0
        # diagnostics
        self.state_sat_steps = 0
        self.inn_sat = 0
        self.stalled = np.zeros(4)
        self.pstalled = np.zeros(4)                # predict steps that rounded to "no change"
        self.n_pred = 0
        self.n_upd = 0
        # remainder-preserving frequency prediction (guard=None | "exact" | g trits)
        guard = 0 if guard is None else int(guard)     # 0 guard trits == plain rounding (D = 1)
        self.guard = guard
        self.D = 3 ** guard
        self.r = 0                                    # integer remainder, units of lsb_f / D
        assert AGE_RANGE == 0.5 and FREQ_RANGE == 20.0
        # exact ratio: freq-lsb units per aging-lsb per step
        c = (Fraction(1, 2) * int(round(dt)) * int(self.lim[1])
             / (Fraction(86400) * 20 * int(self.lim[2])))
        self.cn, self.cd = c.numerator, c.denominator
        self.shift_k = scale3
        self.shift = bool(shift) and scale3 is not None
        if scale3 is not None:                        # aging scale chosen so coupling == 3^-k
            self.lsb[2] = self.lsb[1] * (86400.0 / dt) / 3 ** scale3
            self.cn, self.cd = 1, 3 ** scale3
            assert 0.3 <= self.lim[2] * self.lsb[2] <= 1.0, "aging range off"
        self.sum_acn = 0                              # exact: sum of a*cn*D (ledger)
        self.upd_change = 0                           # freq register change from measurement updates
        self.mismatch = 0                             # float-path vs integer-path disagreements
        self.maxbits = 0                              # widest intermediate integer seen (bits)
        self.intended = 0.0                           # sum of intended freq increments, lsb units
        self.emitted = 0.0                            # sum of increments the register accepted
        self.lost = 0                                 # carries discarded by saturation
        self.sum_units = 0
        self.sum_k = 0
        self.re_sum = 0.0                          # signed innovation rounding error, in lsb
        self.re_n = 0

    def _snap(self, v):
        raw = np.rint(v / self.lsb)
        if np.any(np.abs(raw) > self.lim):
            self.state_sat_steps += 1
        self.q = np.clip(raw, -self.lim, self.lim).astype(np.int64)
        self.x = self.q * self.lsb

    def predict(self, temp_c):
        before = self.q.copy()
        F = self._F(temp_c)
        v = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        # Intended frequency increment from the aging register INTEGER, exact integer arithmetic:
        #   units = nearest( q_age * (cn/cd) * D ),  nearest = floor(x + 1/2)
        q2 = int(self.q[2])
        if self.shift:                                  # c = 3^-k: a digit-position change
            kk, gg = self.shift_k, self.guard
            if gg >= kk:
                units = q2 * 3 ** (gg - kk)             # exact: no digits dropped
                wid = abs(units).bit_length()
            else:
                p = 3 ** (kk - gg)
                units = (q2 + (p - 1) // 2) // p        # drop kk-gg trits: nearest, no ties
                wid = abs(q2).bit_length()
            self.maxbits = max(self.maxbits, wid)
        else:
            num = 2 * q2 * self.cn * self.D + self.cd
            units = num // (2 * self.cd)
            self.maxbits = max(self.maxbits, abs(num).bit_length())
        self.sum_acn += q2 * self.cn * self.D
        # diagnostic only: what the float-derived (v3) path would have produced
        fu = int(np.rint(F[1, 2] * self.x[2] / self.lsb[1] * self.D))
        if fu != units:
            self.mismatch += 1
        self.intended += q2 * self.cn / self.cd
        v[1] = self.x[1]                            # hold the visible register; carry via remainder
        self._snap(v)
        self.sum_units += units
        self.r += units
        k = (self.r + self.D // 2) // self.D        # carry into the visible register
        self.r -= k * self.D
        self.sum_k += k
        lim = int(self.lim[1])
        newq = int(self.q[1]) + k
        clipped = max(-lim, min(lim, newq))
        self.lost += newq - clipped
        self.q[1] = clipped
        self.x[1] = clipped * self.lsb[1]
        self.emitted += float(self.q[1] - before[1])
        self.pstalled += (self.q == before)
        self.n_pred += 1

    def update(self, z):
        resid = z - self.x[0]
        raw = float(np.rint(resid / INN_LSB))
        if abs(raw) > self.inn_lim:
            self.inn_sat += 1
        else:
            self.re_sum += raw - resid / INN_LSB
            self.re_n += 1
        qi = int(np.clip(raw, -self.inn_lim, self.inn_lim))
        self.inn_q = qi
        resid_q = qi * INN_LSB                      # decoded innovation register
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S
        before = self.q.copy()
        self._snap(self.x + K * resid_q)
        self.stalled += (self.q == before)
        self.upd_change += int(self.q[1]) - int(before[1])
        self.n_upd += 1
        self.P = self.P - np.outer(K, self.P[0, :])
        self.last = resid_q
        return resid_q


# ----------------------------------------------------------------- scenarios
SCENARIOS = {
    "baseline (model matches world)": {},
    "quadratic tempco (estimator linear)": {"q2": 0.03},
    "freq step +0.5 ppm at hold+24h": {"step_ppm": 0.5, "step_after_h": 24.0},
    "sensor lag 2h + 1.5C offset": {"sensor_tau": 7200.0, "sensor_off": 1.5},
    "reference glitches (outliers)": {"glitch_p": 0.002},
}

VARIANTS = [
    ("free-run", "free", {}),
    ("phase+freq float", "float", {"full": False}),
    ("full float (unrestricted)", "float", {"full": True}),
    ("full float + innov clip", "float", {"full": True, "clip": half(6) * INN_LSB}),
    ("full ternary 13t/innov 6t", "tern", {"w_state": 13, "w_inn": 6}),
    ("full ternary 13t/innov 13t", "tern", {"w_state": 13, "w_inn": 13}),
    ("full ternary 9t/innov 6t", "tern", {"w_state": 9, "w_inn": 6}),
    ("full ternary 9t+g5/innov 6t", "tern", {"w_state": 9, "w_inn": 6, "guard": 5}),
]


def simulate(seed, scen, variants, days_ref=3.0, days_hold=4.0, dt=60.0, keep=False):
    rng = np.random.default_rng(seed)
    n = int((days_ref + days_hold) * DAY / dt)
    t_loss = days_ref * DAY
    cap_h = days_hold * 24.0

    # hidden truth
    y0 = rng.uniform(1.0, 3.0)
    aging = rng.uniform(0.02, 0.08)
    k_true = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
    rw_step = 0.01 * np.sqrt(dt / DAY)
    q2 = scen.get("q2", 0.0)
    step_ppm = scen.get("step_ppm", 0.0)
    t_step = t_loss + scen.get("step_after_h", 0.0) * 3600.0
    glitch_p = scen.get("glitch_p", 0.0)

    # temperature (true), then what the estimator's sensor reports
    a = np.exp(-dt / 21600.0)
    slow = np.zeros(n)
    for i in range(1, n):
        slow[i] = a * slow[i - 1] + 0.7 * np.sqrt(1 - a * a) * rng.normal()
    t = np.arange(n) * dt
    temp_true = T0 + 3.0 * np.sin(2 * np.pi * t / DAY + rng.uniform(0, 6.28)) + slow
    tau = scen.get("sensor_tau", 0.0)
    if tau > 0:
        b = 1.0 - np.exp(-dt / tau)
        ts = np.zeros(n)
        ts[0] = temp_true[0]
        for i in range(1, n):
            ts[i] = ts[i - 1] + b * (temp_true[i] - ts[i - 1])
    else:
        ts = temp_true
    temp_meas = ts + scen.get("sensor_off", 0.0) + rng.normal(0, 0.05, n)

    filters = {}
    for name, kind, p in variants:
        if kind == "float":
            filters[name] = Discipline(p["full"], dt, p.get("clip"))
        elif kind == "tern":
            filters[name] = TernaryDiscipline(True, dt, **p)
    errs = {name: np.zeros(n) for name, _, _ in variants}

    local = rng.uniform(-5, 5)
    y_rw = 0.0
    tt = np.zeros(n)
    for i in range(n):
        y_rw += rw_step * rng.normal()
        y = (y0 + aging * (t[i] / DAY) + k_true * (temp_true[i] - T0)
             + q2 * (temp_true[i] - T0) ** 2 + y_rw)
        if t[i] >= t_step and step_ppm:
            y += step_ppm
        local += RATE * dt * (1 + y * 1e-6)
        s = int(round((i + 1) * dt))
        tt[i] = s
        ref = RATE * s

        for f in filters.values():
            f.predict(temp_meas[i])
        if s <= t_loss:
            # the counter is latched at the reference edge: an integer reading.
            # Expected reading is the exact rational 59049*s/1600; its fractional
            # part is the known rate-conversion pattern and is subtracted exactly.
            num = 59049 * s
            whole, frac = divmod(num, 1600)
            latched = np.floor(local + rng.normal(0, 0.2))
            z = latched - whole - frac / 1600.0 + 0.5
            if glitch_p and rng.random() < glitch_p:
                z += rng.choice([-1, 1]) * rng.uniform(150, 400)
            for f in filters.values():
                f.update(z)
        for name, kind, _ in variants:
            if kind == "free":
                errs[name][i] = local - ref
            else:
                errs[name][i] = local - ref - filters[name].x[0]

    m = tt > t_loss
    out = {"metrics": {}, "info": {}}
    for name, _, _ in variants:
        e = np.abs(errs[name][m])
        th = tt[m]
        bad = np.nonzero(e > 1.0)[0]
        first = cap_h if len(bad) == 0 else (th[bad[0]] - t_loss) / 3600.0
        out["metrics"][name] = (float(e.max()), float(e[-1]), float(first), len(bad) == 0)
    for name, kind, _ in variants:
        if kind == "tern":
            f = filters[name]
            out["info"][name] = dict(
                inn_sat=f.inn_sat, state_sat=f.state_sat_steps, n_upd=f.n_upd,
                stalled=(f.stalled / max(f.n_upd, 1)).tolist(),
                pstalled=(f.pstalled / max(f.n_pred, 1)).tolist(),
                intended=f.intended, emitted=f.emitted, lost=f.lost, r=f.r,
                mismatch=f.mismatch, maxbits=f.maxbits, n_pred=f.n_pred,
                sum_acn=f.sum_acn, upd_change=f.upd_change, sum_units=f.sum_units,
                sum_k=f.sum_k, D=f.D, cd=f.cd,
                cons_ok=(f.sum_units == f.D * f.sum_k + f.r),
                storage=int(f.widths[0] + f.widths[1] + f.widths[2] + f.widths[3]
                            + (f.guard if isinstance(f.guard, int) else 0)),
                round_err=f.re_sum / max(f.re_n, 1), q=f.q.tolist(),
                widths=f.widths.tolist(), inn_q=f.inn_q, w_inn=f.w_inn,
                face=int(np.floor(local - f.x[0])) % MOD)
    if keep:
        out["tt"] = tt
        out["t_loss"] = t_loss
        out["errs"] = errs
        out["truth"] = dict(k=k_true, aging=aging, y0=y0)
        out["learned"] = {nm: filters[nm].x.copy() for nm in filters}
        out["last_float_innov"] = filters["full float (unrestricted)"].last \
            if "full float (unrestricted)" in filters else None
    return out


def job(args):
    sname, seed, variants, keep = args
    return sname, seed, simulate(seed, SCENARIOS[sname], variants, keep=keep)


def ms(ticks):
    return ticks / RATE * 1000.0


def run_many(scen_names, seeds, variants, workers):
    jobs = [(s, sd, variants, sd == 0) for s in scen_names for sd in seeds]
    res = {s: {} for s in scen_names}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sname, seed, out in ex.map(job, jobs):
            res[sname][seed] = out
    return res


def table(res_s, variants):
    print(f"  {'variant':<28}{'median max err':>16}{'median t>1 tick':>17}{'never >1 tick':>15}")
    for name, _, _ in variants:
        mx = np.median([ms(r["metrics"][name][0]) for r in res_s.values()])
        ft = np.median([r["metrics"][name][2] for r in res_s.values()])
        nv = sum(r["metrics"][name][3] for r in res_s.values())
        print(f"  {name:<28}{mx:>13.1f} ms{ft:>14.1f} h{nv:>9d}/{len(res_s)}")
    a, b = "full ternary 13t/innov 6t", "full float (unrestricted)"
    if a in res_s[0]["metrics"] and b in res_s[0]["metrics"]:
        ratio = np.array([r["metrics"][a][0] / max(r["metrics"][b][0], 1e-9) for r in res_s.values()])
        q1, q2, q3 = np.percentile(ratio, [25, 50, 75])
        print(f"  paired max-error ratio, ternary 13t/6t over float: median {q2:.2f} "
              f"(quartiles {q1:.2f}-{q3:.2f}); ternary lower in {(ratio < 1).sum()}/{len(ratio)} seeds")


def nearest(s):
    """Round a Fraction to the nearest integer (halves up)."""
    return int((s + Fraction(1, 2)).__floor__())


def accumulator_test():
    """Exact-rational test of the aging-increment accumulator, independent of the filter."""
    dt, N = 60, 10080                                 # 7 days of 60 s steps
    cases = {
        "+0.05 ppm/day": lambda n: Fraction(1, 20),
        "-0.05 ppm/day": lambda n: Fraction(-1, 20),
        "+0.05 then -0.05 (sign flip at half)": lambda n: Fraction(1, 20) if n < N // 2 else Fraction(-1, 20),
    }
    print("=== Accumulator test: exact rational reference, 7 days of aging increments ===")
    print("  increment per step = a * 60/86400 ppm (exact fraction); register lsb = 20/((3^w-1)/2) ppm")
    print("  columns: errors in freq-register lsb units, final accumulated value minus exact")
    all_ok = True
    for cname, fn in cases.items():
        print(f"\n  case {cname}")
        print(f"  {'w':>3}{'inc/lsb':>10}{'exact total':>13}{'plain err':>12}{'rem(exact) err':>16}"
              f"{'guard3 err':>12}{'guard5 err':>12}")
        for w in (13, 12, 11, 10, 9, 8):
            lsb = Fraction(20, 1) / half(w)
            deltas = [fn(n) * dt / 86400 / lsb for n in range(N)]   # exact increments in lsb
            exact = sum(deltas)
            # plain: round every step
            qp = sum(nearest(d) for d in deltas)
            # exact remainder: q + r == exact (conservation identity)
            q, r = 0, Fraction(0)
            for d in deltas:
                sv = r + d
                k = nearest(sv)
                q += k
                r = sv - k
            assert q + r == exact and abs(r) <= Fraction(1, 2), "conservation identity failed"
            # integer guard digits: remainder in units of lsb/D
            errs = []
            for g in (3, 5):
                D = 3 ** g
                qg, ru, units_sum, k_sum = 0, 0, 0, 0
                for d in deltas:
                    u = nearest(d * D)
                    units_sum += u
                    ru += u
                    k = (ru + D // 2) // D
                    ru -= k * D
                    qg += k
                    k_sum += k
                assert units_sum == D * k_sum + ru          # exact integer conservation
                assert abs(ru) <= D // 2
                errs.append(float(qg + Fraction(ru, D) - exact))
            inc0 = float(deltas[0])
            print(f"  {w:>3}{inc0:>10.3f}{float(exact):>13.1f}{float(qp - exact):>12.1f}"
                  f"{float(q + r - exact):>16.1f}{errs[0]:>12.2f}{errs[1]:>12.2f}")
    # saturation: the identity must hold once the lost carries are counted
    D, g = 27, 3
    lim, q, ru, lost, units_sum, k_sum = half(3), 0, 0, 0, 0, 0
    for n in range(100):
        u = 19                                              # 19/27 = 0.704 lsb per step
        units_sum += u
        ru += u
        k = (ru + D // 2) // D
        ru -= k * D
        k_sum += k
        nq = q + k
        c = max(-lim, min(lim, nq))
        lost += nq - c
        q = c
    ok = (units_sum == D * k_sum + ru) and (q + lost == k_sum)
    all_ok &= ok
    print(f"\n  Saturation test (3-trit register, +/-{lim}): final q={q}, carries lost={lost}, "
          f"remainder={ru}/{D}; identity  sum(units) = D*(q+lost) + r  holds: {ok}")
    print("  Conservation identity  q + r = q0 + r0 + sum(increment/lsb)  verified exactly with "
          "Fractions in every case above: True")
    print("  (guard-digit error is bounded by N/(2D) lsb from rounding each increment to lsb/D: "
          f"guard3 <= {N / (2 * 27):.0f}, guard5 <= {N / (2 * 243):.0f} lsb)\n")
    return all_ok


def ledger(info):
    """Exact information ledger for the frequency accumulator, in freq-register lsb."""
    D = info["D"]
    quant = Fraction(info["sum_acn"], info["cd"]) - info["sum_units"]      # intended - grid, units
    return dict(quant=float(quant) / D, rem=info["r"] / D, sat=float(info["lost"]),
                upd=float(info["upd_change"]))


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    workers = os.cpu_count() or 1
    seeds = list(range(n_seeds))
    t0 = time.time()
    print(f"Counter: {TRITS} trits, 59049/1600 Hz.  {n_seeds} seeds, {workers} workers, "
          "reference 3 days, holdover 4 days.")
    print("Exhaustive accumulator verification lives in ternary_accum_verify.py (run it first).\n")

    WA = 9                                            # aging and tempco register width

    def kfor(m):
        return 19 - m                                 # k so that the aging range stays ~ +/-0.49 ppm/day

    base = {"w_state": WA, "w_inn": 6}
    variants = [("float", "float", {"full": True})]
    endpoint = [(13, 0), (14, 0), (11, 2), (10, 3), (9, 4), (9, 5)]
    shifted = [(13, 0), (14, 0), (11, 2), (10, 3), (9, 4), (9, 5), (13, 6), (14, 5)]
    twins = [(13, 0), (10, 3), (9, 5), (13, 6)]
    for m, g in endpoint:
        variants.append((f"e{m}+{g}", "tern", dict(base, w_freq=m, guard=g)))
    for m, g in shifted:
        variants.append((f"p{m}+{g}", "tern", dict(base, w_freq=m, guard=g, scale3=kfor(m), shift=True)))
    for m, g in twins:
        variants.append((f"G{m}+{g}", "tern", dict(base, w_freq=m, guard=g, scale3=kfor(m), shift=False)))

    worlds = ["baseline (model matches world)", "quadratic tempco (estimator linear)"]
    res = run_many(worlds, seeds, variants, workers)

    # ---- the shift form must be bit-identical to the generic integer formula at the same scale
    n_eq = 0
    for wname in worlds:
        rs = res[wname]
        for m, g in twins:
            a, b = f"p{m}+{g}", f"G{m}+{g}"
            for sd, r in rs.items():
                assert r["metrics"][a] == r["metrics"][b], (wname, sd, a)
                ia, ib = r["info"][a], r["info"][b]
                assert ia["q"] == ib["q"] and ia["sum_units"] == ib["sum_units"] \
                    and ia["lost"] == ib["lost"] and ia["r"] == ib["r"]
                n_eq += 1
            assert np.array_equal(rs[0]["errs"][a], rs[0]["errs"][b])
    print("=== Shift form vs generic integer formula (same scales) ===")
    print(f"  {n_eq} paired runs ({len(twins)} configurations x {n_seeds} seeds x {len(worlds)} worlds): "
          "final registers, remainder,\n  carry sums, saturation losses and the full seed-0 error "
          "trajectory are BIT-IDENTICAL.  PASS\n")

    allinfo = []
    for wname in worlds:
        rs = res[wname]
        fl = np.median([ms(r["metrics"]["float"][0]) for r in rs.values()])
        print(f"=== {wname} ===  (aging/tempco {WA}t, phase 13t, innovation 6t; float ref {fl:.1f} ms)")
        print("  e = endpoint-fitted scale (v4)   p = power-of-three scale (shift)   "
              "split = main+guard trits on the freq register")
        print(f"  {'variant':>8}{'scale':>9}{'state trits':>13}{'median max err':>16}"
              f"{'median t>1':>12}{'never>1':>9}{'drift kept':>12}{'|quant loss|':>14}{'update dF':>11}")
        names = [f"e{m}+{g}" for m, g in endpoint] + [f"p{m}+{g}" for m, g in shifted]
        for nm in names:
            m, g = (int(x) for x in nm[1:].split("+"))
            mx = np.median([ms(r["metrics"][nm][0]) for r in rs.values()])
            ft = np.median([r["metrics"][nm][2] for r in rs.values()])
            nv = sum(r["metrics"][nm][3] for r in rs.values())
            infos = [r["info"][nm] for r in rs.values()]
            allinfo += infos
            kv = [i["emitted"] / i["intended"] for i in infos if abs(i["intended"]) > 1e-9]
            led = [ledger(i) for i in infos]
            ql = np.median([abs(x["quant"]) for x in led])
            ud = np.median([abs(x["upd"]) for x in led])
            st = infos[0]["storage"]
            print(f"  {nm:>8}{('pow3 k=' + str(kfor(m))) if nm[0] == 'p' else 'endpoint':>9}"
                  f"{st:>13}{mx:>13.1f} ms{ft:>10.1f} h{nv:>6}/{len(rs)}{100 * np.median(kv):>11.0f}%"
                  f"{ql:>12.2f}l{ud:>10.0f}l")
        print("  (quant loss = exact intended minus grid-rounded increment, summed over the run, in freq lsb;\n"
              "   update dF = net freq-register change made by measurement updates, in freq lsb)")
        print("  paired, same split:  endpoint-fitted vs power-of-3 median max error")
        for m, g in endpoint:
            if (m, g) in shifted:
                e = np.median([ms(r["metrics"][f"e{m}+{g}"][0]) for r in rs.values()])
                p = np.median([ms(r["metrics"][f"p{m}+{g}"][0]) for r in rs.values()])
                print(f"    {m}+{g}: {e:6.1f} ms -> {p:6.1f} ms  (ratio {p / e:.2f})")
        print()

    b = res[worlds[0]]
    exact_names = [f"p{m}+{g}" for m, g in shifted if g >= kfor(m)]
    print("=== Ledger entries (baseline world, median over seeds, freq lsb) ===")
    print(f"  {'variant':>8}{'quant loss':>13}{'remainder':>11}{'saturation':>12}{'update dF':>11}")
    for nm in ["e13+0", "e9+5", "p13+0", "p9+5", "p13+6", "p14+5"]:
        led = [ledger(r["info"][nm]) for r in b.values()]
        q_, r_, s_, u_ = (np.median([x[k] for x in led]) for k in ("quant", "rem", "sat", "upd"))
        print(f"  {nm:>8}{q_:>13.3f}{r_:>11.3f}{s_:>12.1f}{u_:>11.0f}")
    for nm in exact_names:
        assert all(ledger(r["info"][nm])["quant"] == 0 for rs in res.values() for r in rs.values()), nm
    print(f"  {', '.join(exact_names)}: increment quantization loss is exactly 0 in every run "
          "(g >= k), as the verifier proved.\n")

    mb_p = max(i["maxbits"] for rs in res.values() for r in rs.values()
               for nm, i in r["info"].items() if nm.startswith("p"))
    mb_e = max(i["maxbits"] for rs in res.values() for r in rs.values()
               for nm, i in r["info"].items() if nm.startswith("e"))
    print(f"  widest intermediate seen:  endpoint-fitted multiply path {mb_e} bits;  shift path {mb_p} bits "
          "(no multiply)")
    print("  conservation identity held in every run:",
          all(i["cons_ok"] for rs in res.values() for r in rs.values() for i in r["info"].values()))
    print(f"\nTotal time {time.time() - t0:.0f} s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, wname in zip(axes, worlds):
            rs = res[wname]
            fl = np.median([ms(r["metrics"]["float"][0]) for r in rs.values()])
            xs = np.arange(len(endpoint))
            ev = [np.median([ms(r["metrics"][f"e{m}+{g}"][0]) for r in rs.values()]) for m, g in endpoint]
            pv = [np.median([ms(r["metrics"][f"p{m}+{g}"][0]) for r in rs.values()]) for m, g in endpoint]
            ax.bar(xs - 0.2, ev, 0.4, color="#ff7f0e", label="endpoint-fitted")
            ax.bar(xs + 0.2, pv, 0.4, color="#17becf", label="power-of-3 scale")
            ax.axhline(fl, color="k", ls="--", lw=0.8, label="float")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{m}+{g}" for m, g in endpoint])
            ax.set_yscale("log")
            ax.set_xlabel("freq register: main + guard trits")
            ax.set_title(wname, fontsize=10)
        axes[0].set_ylabel("median max holdover error (ms)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("ternary_clock_v5.png", dpi=120)
        print("Saved ternary_clock_v5.png")
    except ImportError:
        print("(matplotlib not installed; skipped plot)")


if __name__ == "__main__":
    main()
