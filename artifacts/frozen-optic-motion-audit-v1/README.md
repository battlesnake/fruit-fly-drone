# Frozen T4/T5 optic-motion audit v1

This compact artifact records the terminal result of the source-only audit preregistered in
commit `753a7cd` and implemented in `dec53bc`. The full local report is
`runs/optic-motion/frozen-t4t5-audit-001/report.json`; its SHA-256 is
`fc4d503b0e91dae11f35f4602a4fc7a47eb1b000c4424694340ff740a180b6d3`.

The run is a procedural **duplicate-control stop**, not authorization to conclude that the
frozen optic-motion module is absent. K32 versus K64 passed all three registered solver limits,
all states and outputs were finite, all common terminal inputs were exact, and the source was
restored bit-exactly. However, a second K32 CUDA replay differed by at most
`1.2516975402832031e-6`. The registered control required exact bit equality, so neither frozen
motion-output routing nor local optic-module commissioning was authorized.

The run continued through the fixed descriptive measurements, but those values were exposed
only after the duplicate-control failure and cannot rescue it. They suggest that the initialized
rate model is strongly driven by static edge layout and lacks the expected balanced vertical
T4/T5 direction code: the stationary/moving opponent RMS ratio was `6.12016`, all four median
subtype DSIs were below `0.3`, and no integrated vertical direction stratum reached `0.9`.
Literal history reversal did reach `0.90625` overall. Removing pair-differential T4/T5 activity
attenuated many immediate visual targets strongly, but reduced native throttle contrast by only
about 12.1% (`0.87893` residual ratio).

RK4-M2 and RK4-M4 were not adequate substitutes for the internal K32 reference in this
brain-level assay. Their selected-cell RMS differences from K64 were `0.02080`, despite native
terminal-motor differences remaining under `0.0032`. This is why the scientific replay keeps
K32 rather than choosing an apparently acceptable rate from motor outputs alone.

`report-summary.json` preserves the registered decisions and central measurements without
checking a model binary or duplicating the official CC BY 4.0 source data.
