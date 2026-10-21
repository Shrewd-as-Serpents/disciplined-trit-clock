<!-- ... existing code ... -->
### A. The Ternary Consensus Rotor ($\bmod 27$) & The 3-Fold Triangular Projection
Three autonomous conversational entities—**Lilah, Lily, and Luci**—act as a dynamic 3-trit balanced ternary register (**1 tryte**). In a narrative or biomechanical context (such as a three-headed Cerberus or fractal Hydra sharing a single body), the three entities evaluate stimuli independently but must commit to a single unified physical action at the end of every speaking cycle:
*   **Individual Stances (Trits):** In any given turn cycle, each entity commits to a discrete value $t \in \{-1, 0, +1\}$.
*   **Turn-Order Positional Weighting ($3^0, 3^1, 3^2$) — The Neuromuscular Cascade:** The entities do not hold fixed weights; their weights are determined strictly by the **speaking order** in the current cycle:
    *   **1st Mover ($T_0$):** Weight $3^0 = 1$ (sensory gaze / micro-adjustment).
    *   **2nd Mover ($T_1$):** Weight $3^1 = 3$ (postural coiling / regional offset).
    *   **3rd / Last Mover ($T_2$):** Weight $3^2 = 9$ (motor lunge / regime commitment). Because $9 > 3 + 1$, the Last Mover physically dictates the sign of the final action unless it commits $T_2 = 0$ (planting the body's feet and clamping total movement to $[-4, +4]$).
*   **Signed Consensus Value ($V \in [-13, +13]$):**
    $$V = 9 T_2 + 3 T_1 + 1 T_0$$
    This yields $27$ total states: the zero origin ($V = 0$) and **13 non-zero absolute magnitudes** ($|V| \in \{1, \dots, 13\}$), each existing in positive ($+$) or negative ($-$) polarity.

*   **The 3-Fold Triangle Projection & Hexagonal Lattice ($19$ Distinct Points):**
    Projecting the three entities $(t_{\text{Lilah}}, t_{\text{Lily}}, t_{\text{Luci}})$ onto the vertices of an equilateral triangle at $0^\circ$, $120^\circ$, and $240^\circ$ maps the $27$ tryte states onto **$19$ distinct points on a hexagonal lattice** with integer axial coordinates $(h_x, h_y) = (2t_{\text{Lilah}} - t_{\text{Lily}} - t_{\text{Luci}},\; t_{\text{Lily}} - t_{\text{Luci}})$. Each state splits into two orthogonal channels:
    1.  **Consensus ($s$):** The unweighted trit sum $s = T_2 + T_1 + T_0 \in [-3, +3]$. This directly drives forward/backward progression and defines the $7$ null-sum deadlocks ($s = 0$) versus the $20$ decisive states ($s \neq 0$).
    2.  **Tension ($H$):** The hexagonal displacement vector $(h_x, h_y)$ measuring internal disagreement between the three heads/entities, completely independent of turn order. The squared distance from center is given by the pure integer formula:
        $$\text{Dist}^2 = \frac{1}{2}\sum_{i < j} (t_i - t_j)^2 \in \{0, 1, 3, 4\}$$

*   **Why Only Unison Walks Avoid $\mathcal{Z}_7$:**
    For a 3-turn walk where the Last Mover ($T_2$) sweeps through $\{-1, 0, +1\}$ to stay strictly inside $\mathcal{D}_{20}$, the unweighted sum $s = T_2 + T_1 + T_0$ must never equal $0$. Since $T_2$ takes every value in $\{-1, 0, +1\}$, avoiding $s = 0$ requires $|T_1 + T_0| = 2$, which holds **if and only if** $(T_1, T_0) \in \{(+1, +1), (-1, -1)\}$.

*   **The $7 + 20$ Partition (Defined by Explicit Exclusion):**
<!-- ... existing code ... -->
### C. The Master Ring & CRT Bus ($\bmod 540$)
Because $\gcd(27, 20) = 1$, the Chinese Remainder Theorem guarantees an isomorphism $\mathbb{Z}_{27} \times \mathbb{Z}_{20} \cong \mathbb{Z}_{540}$.
*   **Exact State Reconstruction ($\bmod 540$):**
    Given any ternary state $T = V \bmod 27 \in \{0, \dots, 26\}$ and any vigesimal state $G \in \{0, \dots, 19\}$, the combined system state is a single integer $X \in \{0, \dots, 539\}$:
    $$X = (460 \cdot T + 81 \cdot G) \bmod 540$$
*   **The $\mathbb{Z}_{27}$ Pivot Mirror ($T \to 2P - T$):**
    Because $\{P - 9, P, P + 9\} \bmod 27$ is always an exact $120^\circ$ triad for any pivot state $P \in \mathbb{Z}_{27}$, reflecting any state $T$ across $P$ defines a clean modular involution without requiring a $\bmod 1080$ double cover:
    $$\mathcal{M}_{27}(T;\, P) = (2 P - T) \bmod 27$$
    *(Note: To avoid triggering a $\mathcal{Z}_7$ deadlock during a mirrored triad walk, the reflected state $\text{bal}_{27}(\mathcal{M}_{27}(T; P))$ must also reside in $\mathcal{D}_{20}$.)*

### D. Hyperbolic Ternary Tree Geometry & Fractal Hydra Scaling
A 3-trit tryte is structurally a path down a depth-3 ternary tree where the positional weights $(9, 3, 1)$ correspond to the subtree leaf counts at each level:
*   **Turn Order as Tree Topology:** The entity speaking **last** ($T_2$, weight $9$) makes the top-level root split; the **2nd Mover** ($T_1$, weight $3$) selects the intermediate branch; the **1st Mover** ($T_0$, weight $1$) selects the leaf.
*   **Hyperbolic Tree Distance:** The intrinsic distance between two tryte states $V_a, V_b$ on the decision tree is $2 \times (\text{depth of lowest common ancestor}) \in \{0, 2, 4, 6\}$, rather than linear difference $|V_a - V_b|$. Flipping the Last Mover ($T_2$) traverses the maximum tree distance of $6$, whereas flipping the 1st Mover ($T_0$) traverses a leaf distance of $2$.
*   **Fractal Hydra Expansion ($3^k$ Hierarchy):** In a multi-layered entity (such as a Hydra where each of the $3$ primary heads branches into $3$ sub-heads), each sub-triad evaluates a local tryte whose consensus sign $\text{sgn}(s_i) \in \{-1, 0, +1\}$ feeds into the parent node's trit, scaling the state space fractally while keeping every node's runtime evaluation identical.

---

## 2. The Nervous System: $\tau$-Phase Interference, Elliptic Orbits & Error Syndromes
<!-- ... existing code ... -->