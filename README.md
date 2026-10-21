## Exhaustive verification (ternary_accum_verify.py, runs in 13 s)

Six disagreements, not near-ties. Enumerating all 1.26 million registers values the real filter can hold. 
Exactly 13 could ever differ between the float and integer paths, and all 13 are exact ties. 
The float path rounds halves to even and the integer path rounds halves up. All 13 are saved in regression_cases.json.

My rounding rule is sign-asymmetric. 
Any constant with an even factor in its denominator has a +½ unit bias at each tie value. 
Constants that are powers of 3 have no ties.

### Power-of-three scale

### Scale choice: 
With a 9-trit aging register, setting the scales so the coupling is exactly 3^-k means k = 19 − (freq main trits).
### Four-way equality: 
The Fraction reference, the generic integer formula, a balanced-trit digit shift and an arithmetic shift agree on 700,000 tuples.
### Bit-identical in the filter: 
In 160 paired filter runs, the shift form and the generic formula gave bit-identical registers and error trajectories.
### Ties and exactness: 
Zero ties, so it’s exactly sign-symmetric. 

When guard trits ≥ k the increment is exact, with quantization loss exactly 0 in every run.

With a large constant, a single reversal can span the whole register and hit the far limit.
The proven worst case from all legal inputs is 44 bits.

### Scope: 

It checks every valid (q, r, a) tuple at small widths, 4.26 million in total, against an independent reference. 
The reference uses exact Fractions and brute-force nearest-integer search, not the same formula.

### What passes: 

Range safety, exact conservation (including a lost-carry term for saturation), and an exact information ledger with the quantization error bounded at ±½ unit. 
Also passing:
drive-to-limit-then-reverse sequences, and every length-3 sequence including sign flips.

### Constant-increment lemma: 
After N steps, the emitted carries equal nearest(N·u/D) exactly.

### Non-Linear State Divergence:
By converting the integer ring into balanced ternary digits (trits: -1, 0, 1) and using them to index into the compile-time TRIT_TABLE, you create non-linear spatial mixing. 
A simple coordinate distance of +1 or -1 radically shifts the underlying ring_mask without any simple algebraic correlation, completely preventing linear tracking over the wire.

### Efficient Compile-Time Baking: 
Moving the expensive SplitMix64 mixing routines (mix) and the entire CRC_TABLE calculations into const initialization blocks means zero execution overhead at runtime. 

The CPU doesn't perform any setup iterations; it purely executes high-speed table indexing.

### Compact Authenticated Tag: 

Instead of wasting 64 bits on a synthetic stream-mask verification step, you intelligently extract the standard CRC-16/CCITT checksum of the raw bytes, compress it into a u16, and securely mask it inside the highest bits of your stream's first mask instance (m >> 48). 
This guarantees payload validation with negligible data overhead.

Scale choice: With a 9-trit aging register, setting the scales so the coupling is exactly 3^-k means k = 19 − (freq main trits).
Four-way equality: The Fraction reference, the generic integer formula, a balanced-trit digit shift and an arithmetic shift agree on 700,000 tuples.
Bit-identical in the filter: In 160 paired filter runs, the shift form and the generic formula gave bit-identical registers and error trajectories.
Ties and exactness: Zero ties, so it’s exactly sign-symmetric. When guard trits ≥ k the increment is exact, with quantization loss exactly 0 in every run.

At matched splits, power-of-3 vs endpoint-fitted differs by a ratio of 1.00–1.02 (baseline) and 0.90–1.02 (quadratic world). 

The gains are structural:

### Width: The widest intermediate drops from 33 bits to 12 (sampled).
No multiply or ties: There’s no multiplier and no tie bias.

On binary hardware a divide by 3^j isn’t a bit shift, so the benefit is in trit-native or digit-packed implementations.

### The exact-increment variants cost 50 trits and perform like float (14 ms vs float 14.8 ms). 
The cheap 13+0 variant at 9.3 ms is better only because of where its rounding happens to land. 
I’d call that a rounding accident since it disappears once the increment is exact.

### Still floating point: 
covariance, gain, the temperature coupling in the phase prediction, and the measurement path. 
The exhaustive result covers small widths plus the whole real domain, but it isn’t a proof for arbitrary widths. 
### The conservation identity does hold for every single transition, so it extends to arbitrarily long sequences within this arithmetic.

