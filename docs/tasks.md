# Tasks and temporal populations

VAL/A1 evaluates valuation scenarios; RANK/A2 evaluates rankings with F, T and H
input variants; EVT/B evaluates event direction; FUND/C evaluates fundamental
forecasts. CF/D and CALC/E are diagnostic controls. Category and schema identifiers
remain unchanged in code.

T1/T2/T3 assignments are analysis-side temporal populations determined by pinned
cutoff registries, task dates, guard days and settlement state. Keep them separate;
never send the `time_band` or held-out reference fields to a system. Pending T3
outcomes are not historical resolved labels. The actual dataset/index/cutoff hashes
are part of experiment identity.
