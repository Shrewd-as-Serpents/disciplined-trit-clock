#!/usr/bin/env python3
"""
ternary_accum_verify.py -- exhaustive verification of the bounded integer accumulator that
turns the aging register into frequency-register increments in the disciplined ternary clock.

No Kalman filter, no temperature generator, no random numbers: just the transition function.

One step.  State: main register q (|q| <= L = (3^m-1)/2) and remainder r (|r| <= D//2, D = 3^g).
Input: aging integer a (|a| <= La = (3^wa-1)/2).  Constant c = cn/cd (exact rational) is the
number of frequency-lsb per aging-lsb per step.

    u  = nearest(a*c*D)          increment in lsb/D units, halves toward +infinity
    s  = r + u
    k  = nearest(s/D)            carry into the main register
    r' = s - D*k
    q' = clip(q + k, -L, L)      lost = (q + k) - q'   (excess is discarded, and counted)

Verified for EVERY valid (q, r, a) at each small width, against an independent reference that
uses exact Fractions and brute-force nearest-integer search (not the same formula):

  1  range safety           |q'| <= L and |r'| <= D//2
  2  exact conservation     D*q' + r' + D*lost = D*q + r + u
  3  independent reference  (u, k, r', q', lost) all agree with the brute-force rational model
  4  exact information ledger   D*q + r + x = D*q' + r' + D*lost + e,  x = a*c*D, e = x - u,
                                |e| <= 1/2 unit   (quantization error is the only irreversible loss
                                besides saturation, and it is bounded)
  5  rounding semantics     ties, halves-up, sign symmetry, counted per constant c
  6  saturation             drive into both limits, reverse: no hidden debt
  7  sequences              exhaustive short sequences; constant-increment lemma
  8  power-of-3 scale       c = 3^-k: trit-digit shift == arithmetic shift == generic == reference
plus: classification of every float-vs-integer disagreement over the real filter domain, and the
worst-case intermediate width derived from all legal inputs (not sampled).

Run:  python3 ternary_accum_verify.py
"""
import json
import sys
import time
from fractions import Fraction as Fr


def half(w):
    return (3 ** w - 1) // 2


# ----------------------------------------------------------- independent reference model
def nearest_ref(x):
    """Nearest integer to a Fraction by brute-force search; exact halves go toward +infinity."""
    f = x.__floor__()
    return min((f - 1, f, f + 1, f + 2), key=lambda n: (abs(x - n), -n))


def step_ref(q, r, a, c, D, L):
    """Reference one-step transition, written from the specification with exact Fractions."""
    x = Fr(a) * c * D
    u = nearest_ref(x)
    s = r + u
    k = nearest_ref(Fr(s, D))
    r2 = s - D * k
    nq = q + k
    q2 = L if nq > L else (-L if nq < -L else nq)
    return q2, r2, nq - q2, u, x - u


def step_impl(q, r, a, cn, cd, D, L):
    """The integer-only implementation used by the filter."""
    u = (2 * a * cn * D + cd) // (2 * cd)
    s = r + u
    k = (s + D // 2) // D
    r2 = s - k * D
    nq = q + k
    q2 = max(-L, min(L, nq))
    return q2, r2, nq - q2, u


# ------------------------------------------------------------------- balanced-ternary digits
def trits(n, min_width=0):
    """Balanced-ternary digits of n, least significant first (digits -1, 0, +1)."""
    d = []
    while n != 0:
        r = n % 3
        if r == 2:
            d.append(-1)
            n = (n + 1) // 3
        else:
            d.append(r)
            n //= 3
    d += [0] * (min_width - len(d))
    return d


def untrits(d):
    v = 0
    for t in reversed(d):
        v = v * 3 + t
    return v


for _n in range(-5000, 5001):
    assert untrits(trits(_n)) == _n


# --------------------------------------------------------------- implementations under test
def make_generic(c, D):
    cn, cd = c.numerator, c.denominator

    def f_u(a):
        return (2 * a * cn * D + cd) // (2 * cd)

    def f_split(s):
        k = (s + D // 2) // D
        return k, s - D * k
    return f_u, f_split


def make_trit(k, g, wa):
    """Digit-level: the coupling c = 3^-k is a digit-position change; the remainder split is a
    digit split. Dropping low balanced-ternary digits rounds to nearest by construction."""
    def f_u(a):
        ta = trits(a, wa)
        return untrits([0] * (g - k) + ta) if g >= k else untrits(ta[k - g:])

    def f_split(s):
        ds = trits(s, g)
        return untrits(ds[g:]), untrits(ds[:g])
    return f_u, f_split


def make_shift(k, g):
    """Arithmetic form of the same shift (what the filter runs)."""
    D = 3 ** g
    if g >= k:
        mult = 3 ** (g - k)

        def f_u(a):
            return a * mult
    else:
        p = 3 ** (k - g)
        h = (p - 1) // 2

        def f_u(a):
            return (a + h) // p

    def f_split(s):
        kk = (s + D // 2) // D
        return kk, s - D * kk
    return f_u, f_split


# --------------------------------------------------------------------- exhaustive core check
class Stats:
    def __init__(self):
        self.tuples = 0
        self.pairs = 0
        self.ties = 0
        self.max_abs_e = Fr(0)
        self.lost_nonzero = 0
        self.sat_hi = 0
        self.sat_lo = 0


def verify_core(m, wa, g, c, impls, st):
    L, La, D = half(m), half(wa), 3 ** g
    rmin, rmax = -(D // 2), D // 2
    cn, cd = c.numerator, c.denominator
    for a in range(-La, La + 1):
        x = Fr(a) * c * D
        uref = nearest_ref(x)
        e = x - uref
        assert abs(e) <= Fr(1, 2), ("quantization error out of bound", m, wa, g, c, a)
        if x - x.__floor__() == Fr(1, 2):
            st.ties += 1
        st.max_abs_e = max(st.max_abs_e, abs(e))
        for name, f_u, _ in impls:
            assert f_u(a) == uref, (name, "u", m, wa, g, c, a, f_u(a), uref)
        for r in range(rmin, rmax + 1):
            s = r + uref
            kref = nearest_ref(Fr(s, D))
            r2ref = s - D * kref
            assert rmin <= r2ref <= rmax                                   # 1 range (remainder)
            for name, _, f_split in impls:
                assert f_split(s) == (kref, r2ref), (name, "split", m, wa, g, c, a, r, s)
            st.pairs += 1
            for q in range(-L, L + 1):
                nq = q + kref
                q2 = L if nq > L else (-L if nq < -L else nq)
                lost = nq - q2
                assert -L <= q2 <= L                                       # 1 range (main)
                assert D * q2 + r2ref + D * lost == D * q + r + uref       # 2 conservation
                assert (lost == 0) == (-L <= nq <= L)
                assert q2 + lost == nq
                # 3 agreement of the integer implementation with the Fraction reference
                assert step_impl(q, r, a, cn, cd, D, L) == (q2, r2ref, lost, uref)
                if (q + r + a) % 997 == 0:                                  # full-reference spot check
                    assert step_ref(q, r, a, c, D, L)[:4] == (q2, r2ref, lost, uref)
                st.tuples += 1
                if lost:
                    st.lost_nonzero += 1
                    if lost > 0:
                        st.sat_hi += 1
                    else:
                        st.sat_lo += 1
            # 4 exact ledger in rationals (q cancels): D*q + r + x = D*q' + r' + D*lost + e
            assert (r + x) - (r2ref + e) == D * kref


def fmt_c(c):
    return f"{c.numerator}/{c.denominator}" if c.denominator != 1 else str(c.numerator)


C_LIST = [Fr(1, 2), Fr(3, 2), Fr(1, 3), Fr(1, 9), Fr(1, 27), Fr(2, 3), Fr(5, 7), Fr(17, 5),
          Fr(40, 13), Fr(1, 57600)]
C_BIG = [Fr(1, 2), Fr(1, 9), Fr(3, 2), Fr(40, 13), Fr(1, 57600)]
GEN_CONFIGS = [(2, 2, 0, C_LIST), (2, 2, 1, C_LIST), (3, 2, 2, C_LIST), (3, 3, 1, C_LIST),
               (3, 3, 2, C_LIST), (4, 3, 2, C_LIST), (4, 4, 2, C_LIST), (5, 4, 3, C_BIG)]


def section_generic():
    print("=== 1-4  Exhaustive one-step verification (generic integer formula vs Fraction reference) ===")
    print("  every valid (q, r, a) tuple; constants include exact ties, tiny (dead-zone) and "
          "large (saturating) c\n")
    print(f"  {'m,wa,g':>8}{'constants':>11}{'(a,r) pairs':>14}{'tuples':>12}{'ties':>7}"
          f"{'sat hi/lo':>16}{'max|e| (unit)':>15}  result")
    tot = 0
    for (m, wa, g, cs) in GEN_CONFIGS:
        st = Stats()
        for c in cs:
            verify_core(m, wa, g, c, [("generic", *make_generic(c, 3 ** g))], st)
        tot += st.tuples
        print(f"  {f'{m},{wa},{g}':>8}{len(cs):>11}{st.pairs:>14,}{st.tuples:>12,}{st.ties:>7}"
              f"{f'{st.sat_hi:,}/{st.sat_lo:,}':>16}{float(st.max_abs_e):>15.3f}  PASS")
    print(f"\n  total tuples verified: {tot:,}\n")
    return tot


def section_rounding():
    print("=== 5  Rounding semantics (rule: nearest, exact halves toward +infinity) ===")
    print(f"  {'constant c':>11}{'a values':>10}{'exact ties':>12}{'halves-up OK':>14}"
          f"{'sign-asym (u(a)+u(-a)!=0)':>28}")
    rows = []
    for c in [Fr(1, 2), Fr(3, 2), Fr(5, 7), Fr(17, 5), Fr(1, 3), Fr(1, 9), Fr(1, 27), Fr(2, 3)]:
        D, La = 9, half(4)
        ties = asym = 0
        ok = True
        for a in range(-La, La + 1):
            x = Fr(a) * c * D
            u = nearest_ref(x)
            tie = (x - x.__floor__() == Fr(1, 2))
            if tie:
                ties += 1
                ok &= (u == x.__floor__() + 1)
            ua = nearest_ref(Fr(-a) * c * D)
            if u + ua != 0:
                asym += 1
                assert tie, "asymmetry without a tie"
                assert u + ua == 1
        rows.append((c, ties, asym))
        print(f"  {fmt_c(c):>11}{2 * La + 1:>10}{ties:>12}{str(ok):>14}{asym:>28}")
    print("  -> a tie-bearing c (any c with an even denominator factor) has a built-in +1/2 unit bias "
          "per tie value;\n     c = 3^-k has no ties and is exactly sign-symmetric: no bias.\n")
    return rows


def section_saturation():
    print("=== 6  Saturation: drive into a limit, reverse, check there is no hidden debt ===")
    m, wa, g = 3, 3, 2
    L, La, D = half(m), half(wa), 3 ** g
    cases = span = 0
    for c in [Fr(1, 2), Fr(3, 2), Fr(1, 3), Fr(2, 3), Fr(17, 5), Fr(40, 13), Fr(1, 9)]:
        cn, cd = c.numerator, c.denominator
        for sign in (+1, -1):
            lim = sign * L
            for q0 in range(-L, L + 1):
                for r0 in range(-(D // 2), D // 2 + 1):
                    for a in range(1, La + 1):
                        aa = sign * a
                        if nearest_ref(Fr(aa) * c * D) * sign <= 0:
                            continue
                        q, r, total_lost, hit = q0, r0, 0, False
                        for _ in range(300):
                            q, r, lost, _u = step_impl(q, r, aa, cn, cd, D, L)
                            assert -(D // 2) <= r <= D // 2
                            total_lost += lost
                            if q == lim:
                                hit = True
                                break
                        if not hit:
                            continue
                        for _ in range(4):                       # keep pushing: loss is counted
                            q, r, lost, _u = step_impl(q, r, aa, cn, cd, D, L)
                            assert q == lim and sign * lost >= 0 and -(D // 2) <= r <= D // 2
                        # reverse: the very next step is a function of (q, r) only
                        q1, r1, lost1, u1 = step_impl(q, r, -aa, cn, cd, D, L)
                        q1r, r1r, lost1r, u1r, _e = step_ref(q, r, -aa, c, D, L)
                        assert (q1, r1, lost1, u1) == (q1r, r1r, lost1r, u1r)
                        assert sign * (q1 - lim) <= 0              # it moves off the limit
                        if lost1:                                   # only if the step spans the
                            span += 1                               # whole register
                            assert abs(q + (q1 - q) - 0) <= L and abs(lost1) >= 1
                            assert (q1 == -lim)                     # ... and lands on the far limit
                        cases += 1
    print(f"  {cases:,} drive-to-limit-then-reverse sequences: remainder always inside its window,\n"
          "  every discarded carry counted in `lost`, reversal responds immediately and matches the\n"
          "  reference (policy: excess carries are DISCARDED and counted, not retained)  PASS")
    print(f"  {span:,} of them were large-constant reversals that span the whole register and "
          "saturate at the FAR limit\n  (legitimate, counted in `lost`; an earlier version of "
          "this check wrongly asserted this never happens)\n")
    return cases


def section_sequences():
    print("=== 7  Sequences ===")
    m, wa, g = 3, 3, 2
    L, La, D = half(m), half(wa), 3 ** g
    c = Fr(2, 3)
    cn, cd = c.numerator, c.denominator
    S = [-La, -3, -1, 0, 1, 3, La]
    n = 0
    for q0 in range(-L, L + 1):
        for r0 in range(-(D // 2), D // 2 + 1):
            for seq in [(a, b, d) for a in S for b in S for d in S]:
                qi, ri, qr, rr = q0, r0, q0, r0
                sum_u = sum_lost = 0
                for a in seq:
                    qi, ri, li, ui = step_impl(qi, ri, a, cn, cd, D, L)
                    qr, rr, lr, ur, _e = step_ref(qr, rr, a, c, D, L)
                    assert (qi, ri, li, ui) == (qr, rr, lr, ur)
                    sum_u += ui
                    sum_lost += li
                assert D * qi + ri + D * sum_lost == D * q0 + r0 + sum_u    # ledger over the sequence
                n += 1
    print(f"  {n:,} length-3 sequences (every start state x 343 aging sequences incl. sign flips, "
          "extremes, zeros): composed implementation == reference; sequence ledger exact  PASS")
    # constant-increment lemma: emitted carries after N steps == nearest(N*u/D)
    cases = 0
    big = 10 ** 9
    for g2 in range(0, 4):
        D2 = 3 ** g2
        for u in range(-2 * D2, 2 * D2 + 1):
            q = r = ks = 0
            for N in range(1, 61):
                s = r + u
                k = (s + D2 // 2) // D2
                r = s - k * D2
                q += k
                ks += k
                assert ks == nearest_ref(Fr(N * u, D2)), (g2, u, N)
                assert r == N * u - D2 * ks and abs(r) <= D2 // 2
                cases += 1
    print(f"  constant-increment lemma: after N steps of u units the emitted carries equal "
          f"nearest(N*u/D) exactly,\n  remainder stays within +/-D//2, for {cases:,} (g,u,N) cases  PASS\n")
    return n, cases


def section_shift():
    print("=== 8  Power-of-three scale: c = 3^-k, four independent implementations ===")
    print("  reference (Fractions, brute force)  ==  generic integer formula  ==  balanced-trit digit\n"
          "  shift/split  ==  arithmetic shift, for every (q, r, a)\n")
    print(f"  {'m,wa,g,k':>12}{'(a,r) pairs':>14}{'tuples':>11}{'ties':>6}{'max|e|':>9}  note")
    tot = 0
    for wa in (3, 4):
        for g in range(0, 4):
            for k in range(0, 6):
                m = 3
                c = Fr(1, 3 ** k)
                D = 3 ** g
                impls = [("generic", *make_generic(c, D)), ("trit-digit", *make_trit(k, g, wa)),
                         ("arith-shift", *make_shift(k, g))]
                st = Stats()
                verify_core(m, wa, g, c, impls, st)
                assert st.ties == 0
                tot += st.tuples
                note = "exact (no rounding)" if g >= k else f"drops {k - g} trit(s), nearest"
                if g >= k:
                    assert st.max_abs_e == 0
                print(f"  {f'{m},{wa},{g},{k}':>12}{st.pairs:>14,}{st.tuples:>11,}{st.ties:>6}"
                      f"{float(st.max_abs_e):>9.3f}  {note}")
    print(f"\n  total tuples: {tot:,}.  Zero ties in every case (division by an odd power of 3 cannot "
          "land on a half),\n  so the rule is exactly sign-symmetric, and when g >= k the increment is "
          "EXACT (e = 0).\n")
    return tot


# ------------------------------------------- real filter domain: float vs integer disagreements
def section_real_domain():
    print("=== Float-derived increment vs integer increment over the WHOLE real domain ===")
    print("  every aging register value a in [-La, La] (wa = 9), every (m, g) used by the filter sweeps\n")
    wa = 9
    La = half(wa)
    F12 = 60.0 / 86400.0
    cases = []
    total = 0
    cfgs = 0
    by_class = {"exact tie": 0, "near tie crossed by float": 0, "other": 0}
    for m in range(8, 16):
        Lf = half(m)
        c = Fr(1, 2) * 60 * Lf / (Fr(86400) * 20 * La)
        cn, cd = c.numerator, c.denominator
        lsb1 = 20.0 / Lf
        lsb2 = 0.5 / La
        for g in range(0, 8):
            D = 3 ** g
            cfgs += 1
            for a in range(-La, La + 1):
                total += 1
                ui = (2 * a * cn * D + cd) // (2 * cd)
                x2 = a * lsb2
                fin = F12 * x2 / lsb1 * D
                uf = round(fin)
                if uf != ui:
                    x = Fr(a) * c * D
                    fr = x - x.__floor__()
                    if fr == Fr(1, 2):
                        cls = "exact tie"
                    elif abs(float(fr) - 0.5) < 1e-6:
                        cls = "near tie crossed by float"
                    else:
                        cls = "other"
                    by_class[cls] += 1
                    cases.append(dict(m=m, wa=wa, g=g, a=a, exact=f"{x.numerator}/{x.denominator}",
                                      exact_float=float(x), float_in=repr(fin), float_out=uf,
                                      int_out=ui, cls=cls))
    print(f"  {cfgs} configurations x {2 * La + 1:,} register values = {total:,} cases enumerated")
    print(f"  disagreements: {len(cases)}   "
          + "   ".join(f"{k}: {v}" for k, v in by_class.items()))
    for cs in cases[:12]:
        print(f"    m={cs['m']:>2} g={cs['g']} a={cs['a']:>6}  exact={cs['exact_float']:.9f}  "
              f"float path -> {cs['float_out']}   integer -> {cs['int_out']}   [{cs['cls']}]")
    if len(cases) > 12:
        print(f"    ... {len(cases) - 12} more in regression_cases.json")
    with open("regression_cases.json", "w") as fh:
        json.dump(cases, fh, indent=1)
    print("  every case saved to regression_cases.json as a permanent regression test")
    print("  (float path uses Python round(), i.e. ties-to-even; the integer path rounds halves up)\n")
    return cases, total


def section_widths():
    print("=== Worst-case intermediate width from ALL legal inputs (not sampled) ===")
    print("  numerator 2*a*cn*D + cd at a = +/-La, aging register 9 trits\n")
    print(f"  {'freq main m':>12}" + "".join(f"{'g=' + str(g):>9}" for g in (0, 3, 5, 6)) + "   (bits)")
    La = half(9)
    worst = 0
    for m in range(8, 16):
        Lf = half(m)
        c = Fr(1, 2) * 60 * Lf / (Fr(86400) * 20 * La)
        cn, cd = c.numerator, c.denominator
        row = []
        for g in (0, 3, 5, 6):
            D = 3 ** g
            w = max(abs(2 * a * cn * D + cd).bit_length() for a in (-La, La))
            row.append(w)
            worst = max(worst, w)
        print(f"  {m:>12}" + "".join(f"{w:>9}" for w in row))
    print(f"\n  worst case over the table: {worst} bits (~{worst / 1.585:.0f} trits), "
          "for the generic endpoint-fitted constant.")
    print("  With c = 3^-k the increment is a digit shift: widest operand is the aging register "
          "itself (9 trits)\n  shifted left by at most g-k digits; there is no multiplier.\n")
    return worst


def main():
    t0 = time.time()
    r = {}
    r["generic_tuples"] = section_generic()
    section_rounding()
    section_saturation()
    section_sequences()
    r["shift_tuples"] = section_shift()
    cases, total = section_real_domain()
    section_widths()
    print(f"ALL CHECKS PASSED.  {r['generic_tuples'] + r['shift_tuples']:,} exhaustive transition tuples, "
          f"{total:,} real-domain cases enumerated, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
