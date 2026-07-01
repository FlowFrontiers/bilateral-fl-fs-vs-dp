# Data

This directory contains two **IP-deidentified** home-router flow parquet files, shipped with the repository via **git LFS**.

> **Release note.** The direct IP address fields `src_ip` and `dst_ip` have been removed. The released files contain per-flow statistics, protocol/label metadata, and the columns the pipeline consumes; **no public/server IP addresses remain**. The files still include `src_mac` and `dst_mac` columns because they are part of the exported flow schema, although the FL pipeline does not consume them. The raw, IP-bearing records are retained privately by the authors and are not distributed. Column values are otherwise byte-faithful to the originals (dtypes are losslessly downcast; the FL pipeline reproduces identical results).

Fetching the data requires git LFS:

```bash
git lfs install
git clone <repo-url>     # data is fetched automatically
# or, in an existing clone:  git lfs pull
```

## Files

```
data/
├── home_A.parquet        # ~111 MB, Home A (11.9 days, Feb 11–23 2026)
└── home_B.parquet        #  ~63 MB, Home B extended (11.5 days, Feb 24–Mar 8 2026)
```

Verify integrity against `repro_manifest.json`:

```bash
shasum -a 256 data/home_A.parquet data/home_B.parquet
# Expected:
# e34b01a1ac786e8decda51336c81d8163ace4cf5fbe01deea5612b727f510bcb  data/home_A.parquet
# b46b21c1d6b7b34ffd2028bff174f130940e09e35dd1c84ede8cdc4574698a43  data/home_B.parquet
```

## Schema

Each parquet file has **78 columns** extracted by a custom flow metering tool backed by [nDPI 5.0](https://www.ntop.org/products/deep-packet-inspection/ndpi/), with `src_ip` and `dst_ip` removed for privacy. Dtypes are losslessly downcast to the smallest exact type. The columns include:

| Group | Columns | Types |
|-------|---------|-------|
| Flow identifiers | `src_port`, `dst_port`, `protocol`, `src_mac`, `dst_mac` | int / str |
| Timestamps | `bidirectional_{first,last}_seen_ms`, `{src2dst,dst2src}_{first,last}_seen_ms` | int (epoch ms) |
| Duration | `bidirectional_duration_ms`, `{src2dst,dst2src}_duration_ms` | int |
| Packet/byte counts | `bidirectional_{packets,bytes}`, `{src2dst,dst2src}_{packets,bytes}` | int |
| DPI metadata | `pkts_to_classify`, `pkts_to_metadata` | int |
| TCP flags | `bidirectional_{syn,cwr,ece,urg,ack,psh,rst,fin}_packets` (+ per-direction) | int |
| Packet size stats | `bidirectional_ps_{min,max,mean,stddev}` (+ per-direction) | int / float |
| Inter-arrival time | `bidirectional_piat_{min,max,mean,stddev}` (+ per-direction) | int / float |
| Labels | `label` (fine-grained app), `confidence` (DPI method), `category` (6 classes) | str |
| TLS/QUIC | `tls_version`, `quic_version` | str |
| SPLT | `splt_dir`, `splt_ps`, `splt_piat` | list |

## Preprocessing Assumptions

The **FL pipeline** (`fl_pipeline/data.py`) applies three runtime filters:

1. **Minimum packet count**: `bidirectional_packets >= 2`. Single-packet flows produce all-zero values for duration, stddev, and PIAT features.
2. **DPI-only ground truth**: `confidence == "DPI"`. Ensures labels come from full deep packet inspection, not port/IP matching or partial heuristics.
3. **CORE class selection**: Keeps 6 categories satisfying `>0.4%` share AND `>1000` flows in each home. Excludes "Unspecified".

After all filters (with canonical extended Home B): **996,450 flows (Home A) + 619,284 flows (Home B) = 1,615,734 total**.

The 16 features used for model training are defined in `fl_pipeline/config.py`.
