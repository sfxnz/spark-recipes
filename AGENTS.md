# AGENTS.md — spark-recipes catalog

This repo is the index, not a serve. One README table. Each model is its own GitHub repo with `run.sh` / `stop.sh`. Do not turn this catalog into a parent git of those repos.

[L.A.I.L](https://github.com/sfxnz/L.A.I.L) is the serve/eval console that produced the Nemotron numbers. It is not a recipe. Do not mix autoconfig overlays with these cookbooks.

## Working rules

- Every number in the table is copied from that recipe's README. A cell is `—` where the README does not state the value. Do not invent, round across harnesses, or paste L.A.I.L UI numbers into a 2× Spark prose/structured cell.
- "Last verified" is the date stated in the README (Nemotron run id, Ornith "19 Aug 2026"), otherwise the last commit on `main`.
- Nemotron `~93 tok/s` is the L.A.I.L lab bench (`Decode (c=1, median)`, run id `20260812T131332Z_d0f7ae`), not `bench_decode.py`.
- Ornith publishes `vllm bench serve` output throughput per command. Do not treat any one tok/s as a general decode number. Its max-concurrency 1 run reported peak concurrent 2.
- Adding a recipe means a new row plus a link. Keep the methodology, hardware, and refuse-guard notes in this README in sync with what the linked repos actually do.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.

## Shared facts for a new recipe repo

2× Spark: QSFP RoCE `10.100.8.1` / `10.100.8.2`, TP=2, head spark1, worker spark2. Pin `NCCL_IB_HCA`. This lab uses `enp1s0f1np1` / `rocep1s0f1`. Unpinned NCCL picks a DOWN HCA.

1× Spark: one GB10, 121Gi unified memory per `free -h`.

Change one knob at a time against the frozen table. Revert when it does not beat noise. `run.sh` should refuse settings that were measured to fail. `FORCE_UNSAFE_*=1` overrides.

An `AGENTS.md` in this catalog does not apply inside the per-model repos. Put recipe-specific rules in that recipe's own `AGENTS.md`.
