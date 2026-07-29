# TelecomTS Snapshot

`telecomts_800_even.jsonl.gz` is the exact 800-row snapshot used by the
paper. It retains only `KPIs` and `labels`, the two source fields consumed by
the benchmark. No model target or prediction is stored in the snapshot.

The rows were sampled as eight evenly spaced pages of 100 records from the
32,000-row `train` split exposed by the Hugging Face datasets server. The
unmodified source dataset is `AliMaatouk/TelecomTS`, released under the MIT
license. See `manifest.json` for provenance and the SHA-256 checksum.

The benchmark divides each 128-step record into:

- input: the first 96 KPI steps;
- proxy latency target: a deterministic projection of the final 32 steps;
- violation target: the supplied anomaly and congestion labels.

TelecomTS does not expose native AMF, SMF, or UPF telemetry. This experiment
therefore evaluates a documented 5G observability-to-5GC projection, not a
claim that the source records are direct 5G Core measurements.
