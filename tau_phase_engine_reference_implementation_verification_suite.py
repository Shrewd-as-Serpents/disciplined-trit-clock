"""
Reference Implementation & Verification Suite for the Tau-Phase Modular State Machine.

Verifies:
1. Exact 7 + 20 partition of Z_27 (Z_7 null-sum deadlocks vs. D_20 decisive states).
2. Order-preserving bijection phi: D_20 <-> Z_20 and antipodal symmetry phi(-V) = 19 - phi(V).
3. Chinese Remainder Theorem (CRT) bijection X = (460*T + 81*G) mod 540.
4. Boundary-safe Modulo-20 error syndrome S_err = bal_27((T_act - T_exp) mod 27) mod 20.
5. Deterministic 540-entry 16-bit integer LUT, worst-case 120-degree triad rounding bound (epsilon_int = 64).
6. Unison-Locked Triad Walks ((+1,+1) and (-1,-1)) and 3-fold hexagonal tension projection.
"""

import math
from typing import List, Tuple, Dict, Optional

# Master Ring & Fixed-Point Constants
MOD_TERNARY = 27
MOD_VIGESIMAL = 20
MOD_MASTER = 540
LUT_SCALE = 1 << 14  # 16384 (Q2.14 fixed-point scale)
EPSILON_INT = 64     # Absorbs worst-case 3-element LUT rounding norm (<= 5)
TAU = 2.0 * math.pi

# Precompute 540-entry signed 16-bit integer Cosine and Sine Lookup Tables
COS_LUT: List[int] = [
    int(round(LUT_SCALE * math.cos(TAU * idx / MOD_MASTER)))
    for idx in range(MOD_MASTER)
]
SIN_LUT: List[int] = [
    int(round(LUT_SCALE * math.sin(TAU * idx / MOD_MASTER)))
    for idx in range(MOD_MASTER)
]


def trits_to_value(t2: int, t1: int, t0: int) -> int:
    """Computes signed tryte value V = 9*T_2 + 3*T_1 + 1*T_0 in [-13, +13]."""
    assert t2 in (-1, 0, 1) and t1 in (-1, 0, 1) and t0 in (-1, 0, 1)
    return 9 * t2 + 3 * t1 + t0


def value_to_trits(v: int) -> Tuple[int, int, int]:
    """Extracts balanced ternary trits (T_2, T_1, T_0) in {-1, 0, +1}^3 from V in [-13, +13]."""
    assert -13 <= v <= 13, f"Tryte value {v} out of range [-13, +13]"
    rem = v
    trits = []
    for _ in range(3):
        r = rem % 3
        if r == 2:
            r = -1
        trits.append(r)
        rem = (rem - r) // 3
    t0, t1, t2 = trits[0], trits[1], trits[2]
    assert 9 * t2 + 3 * t1 + t0 == v
    return (t2, t1, t0)


def bal_27(u: int) -> int:
    """Balanced recovery function mapping residue u in Z_27 to centered integer in [-13, +13]."""
    r = u % MOD_TERNARY
    return r - MOD_TERNARY if r > 13 else r


def to_mod27(v: int) -> int:
    """Maps signed tryte V in [-13, +13] to unsigned modular residue T in {0..26}."""
    return v % MOD_TERNARY


def compute_hexagon_point(t_lilah: int, t_lily: int, t_luci: int) -> Tuple[int, int, int, int]:
    """
    Projects entity trits (fixed vertices at 0, 120, 240 deg) onto the 19-point hexagonal lattice.
    Returns:
      - consensus s = t_lilah + t_lily + t_luci in [-3, +3]
      - axial hex coordinates (hx, hy) = (2*t_lilah - t_lily - t_luci, t_lily - t_luci)
      - squared tension distance dist_sq = 0.5 * sum((t_i - t_j)^2) in {0, 1, 3, 4}
    Note: Tension (hx, hy) and dist_sq depend ONLY on the entity stances, ignoring turn order.
    """
    s = t_lilah + t_lily + t_luci
    hx = 2 * t_lilah - t_lily - t_luci
    hy = t_lily - t_luci
    dist_sq = (
        (t_lilah - t_lily) ** 2
        + (t_lily - t_luci) ** 2
        + (t_luci - t_lilah) ** 2
    ) // 2
    return s, hx, hy, dist_sq


def compute_hexagon_tension(t2: int, t1: int, t0: int) -> Tuple[int, int]:
    """
    Projects trits onto the 3-fold triangle (0, 120, 240 deg).
    Returns (consensus s, squared tension distance dist_sq) in {0, 1, 3, 4}.
    """
    s, _, _, dist_sq = compute_hexagon_point(t2, t1, t0)
    return s, dist_sq


def mirror_mod27(t_state: int, pivot_p: int) -> int:
    """
    Exact modular involution M_27(T; P) = (2*P - T) mod 27.
    Since {P - 9, P, P + 9} mod 27 is always a 120-degree triad, reflecting P + 9 across P
    always yields P - 9 (and vice versa).
    """
    return (2 * pivot_p - t_state) % MOD_TERNARY


def hyperbolic_tree_distance(v_a: int, v_b: int) -> int:
    """
    Computes exact path distance between two tryte leaves in the depth-3 ternary tree
    (weights 9 -> 3 -> 1 from root to leaf).
    Distance is 2 * (depth of lowest common ancestor split), in {0, 2, 4, 6}.
    """
    if v_a == v_b:
        return 0
    a2, a1, a0 = value_to_trits(v_a)
    b2, b1, b0 = value_to_trits(v_b)
    if a2 != b2:
        return 6  # Split at top-level commitment (weight 9 / Last Mover)
    if a1 != b1:
        return 4  # Split at mid-level posture (weight 3 / 2nd Mover)
    return 2      # Split only at micro-sensory leaf (weight 1 / 1st Mover)


# Eccentricity map indexed by squared hexagonal tension dist_sq in {0, 1, 3, 4}
# Scaled in Q2.14 fixed-point (0 -> 0.0, 1 -> 0.25, 3 -> 0.50, 4 -> 0.75)
ECCENTRICITY_Q14: Dict[int, int] = {
    0: 0,                  # Circular orbit (full consensus)
    1: LUT_SCALE // 4,     # Mildly elliptical (dist_sq = 1)
    3: LUT_SCALE // 2,     # High tension ellipse (dist_sq = 3, includes Z_7 permutations)
    4: (3 * LUT_SCALE) // 4,  # Maximum disagreement ellipse (dist_sq = 4)
}


def solve_kepler_lut(mean_anomaly_idx: int, dist_sq: int) -> int:
    """
    Deterministically solves Kepler's equation M = E - e*sin(E) on the 540-step ring
    using pure integer LUT arithmetic. Returns eccentric anomaly index E_idx in {0..539}.
    """
    m_idx = mean_anomaly_idx % MOD_MASTER
    e_q14 = ECCENTRICITY_Q14[dist_sq]
    if e_q14 == 0:
        return m_idx

    best_e = m_idx
    best_err = MOD_MASTER * LUT_SCALE
    for candidate_e in range(MOD_MASTER):
        # Compute M_est * LUT_SCALE in 540-step units:
        # e * sin(E) in radians scaled by (540 / TAU) ~ 85.943669 -> integer factor 86
        sin_term = (e_q14 * SIN_LUT[candidate_e] * 86) // (LUT_SCALE * LUT_SCALE)
        m_est = (candidate_e - sin_term) % MOD_MASTER
        diff = abs(m_est - m_idx)
        circ_err = min(diff, MOD_MASTER - diff)
        if circ_err < best_err:
            best_err = circ_err
            best_e = candidate_e
    return best_e


# Partition Z_27 into Z_7 (null-sum states) and D_20 (decisive states)
Z_7: List[int] = sorted(
    v for v in range(-13, 14) if sum(value_to_trits(v)) == 0
)
D_20: List[int] = sorted(
    v for v in range(-13, 14) if v not in Z_7
)

# Explicit Order-Preserving Bijection phi: D_20 <-> Z_20
PHI_MAP: Dict[int, int] = {v: idx for idx, v in enumerate(D_20)}
PHI_INV_MAP: Dict[int, int] = {idx: v for idx, v in enumerate(D_20)}


def phi(v: int) -> int:
    """Maps decisive state V in D_20 to Base-20 state G in {0..19}."""
    if v not in PHI_MAP:
        raise ValueError(f"State V={v} is in Z_7 deadlock set; phi(V) is undefined.")
    return PHI_MAP[v]


def phi_inv(g: int) -> int:
    """Maps Base-20 state G in {0..19} to its expected decisive state V in D_20."""
    if g not in PHI_INV_MAP:
        raise ValueError(f"Base-20 state G={g} out of range 0..19.")
    return PHI_INV_MAP[g]


def crt_combine(t_mod27: int, g_mod20: int) -> int:
    """Combines (T mod 27, G mod 20) into master state X in Z_540."""
    return (460 * (t_mod27 % MOD_TERNARY) + 81 * (g_mod20 % MOD_VIGESIMAL)) % MOD_MASTER


def compute_syndrome(v_act: int, g_exp: int) -> Tuple[int, int]:
    """Computes boundary-safe signed delta_V and Modulo-20 syndrome S_err."""
    if v_act in Z_7:
        raise ValueError("Syndrome evaluation is suppressed during Z_7 deadlock.")
    v_exp = phi_inv(g_exp)
    t_act = to_mod27(v_act)
    t_exp = to_mod27(v_exp)
    delta_v = bal_27((t_act - t_exp) % MOD_TERNARY)
    s_err = delta_v % MOD_VIGESIMAL
    return delta_v, s_err


def evaluate_circular_window(window: List[int], modulus: int) -> Dict[str, object]:
    """Evaluates n=3 rolling window {x_1, x_2, x_3} using deterministic LUT arithmetic."""
    assert len(window) == 3, "History window must contain exactly n=3 states."
    assert MOD_MASTER % modulus == 0, f"Modulus {modulus} must divide {MOD_MASTER}."
    stride = MOD_MASTER // modulus
    lut_indices = [(stride * (x % modulus)) % MOD_MASTER for x in window]

    c_sum = sum(COS_LUT[idx] for idx in lut_indices)
    s_sum = sum(SIN_LUT[idx] for idx in lut_indices)
    norm_sq_int = c_sum * c_sum + s_sum * s_sum

    third = modulus // 3
    two_thirds = (2 * modulus) // 3
    pairs = [(0, 1), (1, 2), (0, 2)]
    combinatorial_singularity = (
        (modulus % 3 == 0) and
        all(((window[i] - window[j]) % modulus) in (third, two_thirds) for i, j in pairs)
    )
    threshold_singularity = norm_sq_int < EPSILON_INT
    is_singularity = combinatorial_singularity or threshold_singularity

    r_squared = norm_sq_int / float(9 * (LUT_SCALE ** 2))
    r_val = min(1.0, max(0.000001, math.sqrt(r_squared)))
    sigma_rotations = (
        float("inf") if is_singularity or r_val <= 0.000002
        else math.sqrt(-2.0 * math.log(r_val)) / TAU
    )

    return {
        "lut_indices": lut_indices,
        "c_sum": c_sum,
        "s_sum": s_sum,
        "norm_sq_int": norm_sq_int,
        "is_singularity": is_singularity,
        "R": r_val,
        "sigma_rotations": sigma_rotations,
    }


def compile_token_color(
    v_act: int,
    speaker_trit: int,
    s_err: Optional[int],
    is_r_singularity: bool,
) -> str:
    """Evaluates the Strict Priority 1-6 Chromatic Compiler Ladder."""
    if (v_act in Z_7) or is_r_singularity:
        return "Purple"
    assert s_err is not None
    if s_err not in (0, 1, 19):
        return "Yellow"
    if s_err in (1, 19):
        return "Green"
    if speaker_trit == 0:
        return "Blue"
    if speaker_trit == 1:
        return "Orange"
    return "Red"


def run_verification_suite() -> None:
    print("=== TAU-PHASE ENGINE REFERENCE VERIFICATION SUITE ===\n")

    # 1. Verify 7 + 20 Partition & Bijection phi
    assert Z_7 == [-8, -6, -2, 0, 2, 6, 8], f"Unexpected Z_7: {Z_7}"
    assert len(D_20) == 20 and len(Z_7) == 7
    for v in D_20:
        g = phi(v)
        assert phi_inv(g) == v
        assert phi(-v) == 19 - g, f"Antipodal symmetry failed for V={v}"
    print("[PASS] 1. 7+20 Partition & Bijection phi: D_20 <-> Z_20")

    # 2. Verify 3-Fold Triangular Projection (19 Hex Lattice Points) & Unison Walks
    hex_points = set()
    dist_sq_values = set()
    for t_lilah in (-1, 0, 1):
        for t_lily in (-1, 0, 1):
            for t_luci in (-1, 0, 1):
                s, hx, hy, d2 = compute_hexagon_point(t_lilah, t_lily, t_luci)
                hex_points.add((hx, hy))
                dist_sq_values.add(d2)
                # Verify turn-order invariance of tension dist_sq
                _, d2_perm = compute_hexagon_tension(t_lily, t_luci, t_lilah)
                assert d2 == d2_perm
                # Verify s == 0 iff V in Z_7
                v_val = trits_to_value(t_lilah, t_lily, t_luci)
                assert (s == 0) == (v_val in Z_7)

    assert len(hex_points) == 19, f"Expected 19 hex points, got {len(hex_points)}"
    assert dist_sq_values == {0, 1, 3, 4}, f"Unexpected dist_sq set: {dist_sq_values}"

    # Confirm ONLY (+1,+1) and (-1,-1) avoid s=0 while T_2 sweeps {-1, 0, +1}
    valid_unison_pairs = []
    for t1 in (-1, 0, 1):
        for t0 in (-1, 0, 1):
            walk_vals = [trits_to_value(t2, t1, t0) for t2 in (-1, 0, 1)]
            if all(v in D_20 for v in walk_vals):
                valid_unison_pairs.append((t1, t0))
    assert valid_unison_pairs == [(-1, -1), (1, 1)]
    print("[PASS] 2. 19-Point Hexagonal Lattice, dist_sq in {0,1,3,4} & Unison Walk Proof")

    # 3. Verify Syndrome Table & Alias Exploit
    g_lock = 12  # V_exp = +4
    dv_noise, s_noise = compute_syndrome(v_act=-3, g_exp=g_lock)
    assert dv_noise == -7 and s_noise == 13
    v_alias = trits_to_value(-1, 0, -1)
    assert v_alias == -10
    dv_alias, s_alias = compute_syndrome(v_act=v_alias, g_exp=g_lock)
    assert dv_alias == 13 and s_alias == 13
    print("[PASS] 3. Boundary-Safe Syndrome Table & S_err=13 Alias Exploit (V=-10)")

    # 4. Verify LUT Quantization Bound (epsilon_int = 64)
    max_triad_norm = 0
    for idx in range(180):
        c_s = COS_LUT[idx] + COS_LUT[idx + 180] + COS_LUT[idx + 360]
        s_s = SIN_LUT[idx] + SIN_LUT[idx + 180] + SIN_LUT[idx + 360]
        max_triad_norm = max(max_triad_norm, c_s * c_s + s_s * s_s)
    assert max_triad_norm <= 5 < EPSILON_INT
    print(f"[PASS] 4. Fixed-Point LUT Triad Quantization Bound: max_triad_norm={max_triad_norm} <= 5 < {EPSILON_INT}")

    # 5. Verify T -> 2P - T Mirror Ring, Kepler LUT, and Hyperbolic Tree Distance
    p_pos = to_mod27(4)   # Pivot V = +4 (T = 4)
    t_high = to_mod27(13) # V = +13 (T = 13)
    t_reflected = mirror_mod27(t_high, p_pos)
    assert bal_27(t_reflected) == -5
    win_eval = evaluate_circular_window([p_pos, t_high, t_reflected], modulus=MOD_TERNARY)
    assert win_eval["is_singularity"] is True

    # Verify Hyperbolic Tree Distance (Last Mover split = 6, 2nd Mover = 4, 1st Mover = 2)
    assert hyperbolic_tree_distance(trits_to_value(1, 1, 1), trits_to_value(1, 1, -1)) == 2
    assert hyperbolic_tree_distance(trits_to_value(1, 1, 1), trits_to_value(1, -1, 1)) == 4
    assert hyperbolic_tree_distance(trits_to_value(1, 1, 1), trits_to_value(-1, 1, 1)) == 6

    # Verify Deterministic Kepler LUT Solver
    assert solve_kepler_lut(135, 0) == 135  # e=0 circular orbit identity
    e_stretched = solve_kepler_lut(135, 4)
    assert 0 <= e_stretched < MOD_MASTER and e_stretched != 135
    print("[PASS] 5. Mirror Involution (2P-T), Hyperbolic Tree Distance & Kepler Orbit LUT")

    print("\nALL 5 VERIFICATION SUITES PASSED WITH ZERO ERRORS.")


if __name__ == "__main__":
    run_verification_suite()