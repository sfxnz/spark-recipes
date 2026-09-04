# spark-recipes

Recipes for running open-weight models on 1× and 2× NVIDIA DGX Spark (GB10) with vLLM. Each recipe is its own repo with `run.sh` / `stop.sh`, a pinned image, a model snapshot pin where the README states one, and a table measured in this lab. The 2× Spark recipes also carry refuse-guards in `run.sh` for settings that were measured to fail on this hardware, a frozen decode benchmark (`bench_decode.py`), agent-readiness probes (tool-call, multi-turn, thinking-off, long-context needle), and published evidence. Every number below is copied from the recipe's README. A cell is `—` where the README does not state the value.

## Recipes

| Model | Hardware | Engine + image pin | Quant | Max context | Decode tok/s c=1 (prose) | Decode tok/s c=2 (prose) | Spec decode | Notes | Last verified | Link |
|---|---|---|---|---|---:|---:|---|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp (284B / 13B active + 32-layer ViT) | 2× Spark, TP=2 | vLLM (B12X) · `dsv4-flash-vision-sm121` built from `eugr/spark-vllm-b12x@sha256:7dc02f16…` · snapshot `86f746b` | MXFP4 experts, FP8 attention | 1,048,576 | 26.2 | 20.5 (agg 40.5) | DSpark-6 | Vision wired (`image_url`, `smoke_vision.py`). 1M window on a 12 GiB fp8 KV pin (pool 1,250,741 tokens). Needles: 13,349 and 53,349 tokens hit, 200,013 missed, 1M not run. Thinking off by default. `run.sh` refuses a KV pin below 12 GiB above a 289,024 window; do not raise `MAX_NUM_SEQS` on this pin. | 2026-09-02 | [repo](https://github.com/sfxnz/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark) |
| GLM-5.3-Flash-NVFP4 (320B / 18B active) | 2× Spark, TP=2 | vLLM · local `glm53-sm121-v11` chain built from `vllm/vllm-openai:glm53-flash-arm64-cu130` · checkpoint `caca4e6` | NVFP4 weight-only on routed experts | 327,680 (native 1,048,576) | 21.2 | 16.6 (agg 33.2) | DFlash2-7 (`SPEC=mtp` for MTP-4) | 318,123-token needle (97% of window) answered exactly. `chat_template.jinja` fixes the thinking-off CoT leak. `run.sh` refuses > 327,680 on fp8 KV and any MoE backend other than marlin (`flashinfer_cutlass` OOM'd spark2). | 2026-09-02 | [repo](https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark) |
| Qwen3.8-Flash-Next-NVFP4 | 2× Spark, TP=2 | vLLM · `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188f…` plus `docker/ple_layer.py` overlay · snapshot `7b719225` | ModelOpt NVFP4 W4A4 experts, FP8 PLE | 1,048,576 (native 262,144) | 39.5 | 34.9 (agg 64.4) | MTP-3 | Vision wired (`image_url`, `smoke_vision.py`). Thinking off by default (`enable_thinking=false`). Occupancy is eight sequences. 1M is a lab ceiling, not a trained window. Auto MoE selected `FLASHINFER_CUTLASS`. `run.sh` refuses a window above 1,048,576, seqs above 8, global `marlin` (unquantized MTP rejects it), and `flashinfer_cutlass`. | 2026-09-04 | [repo](https://github.com/sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark) |
| Nemotron 3.5 Lightning 30B-A3B NVFP4 | 1× Spark | vLLM · `vllm/vllm-openai:v0.27.1` (stock) | NVFP4 (`modelopt_mixed`) | 262,144 | ~93 ¹ | — | DSpark, 3 draft tokens | Hybrid Mamba-2 + MoE. Prefill ~678 tok/s, TTFT p50 ~246 ms, mixed agent bench 20/20 OK. `gpu-memory-utilization 0.8`. | 2026-08-12 | [repo](https://github.com/sfxnz/Nemotron-3.5-Lightning-vLLM-DGX-Spark) |
| Ornith-1.5-35B-A3B-NVFP4 | 1× Spark, TP=1 | vLLM · `vllm/vllm-openai:v0.27.1` @ `sha256:0a51ea5b…` · snapshot `9660379` | ModelOpt W4A16_NVFP4 experts + FP8 attention | 262,144 | — ² | — ² | MTP-3 | `vllm bench serve` tables only: 52.61 tok/s output at max-concurrency 1 (completions 256/128), 89.88 at max-concurrency 2 (completions 256/512). Thinking off required for chat benches. `gpu-memory-utilization 0.4`; 59Gi used / 121Gi total after load. | 2026-08-19 | [repo](https://github.com/sfxnz/Ornith-1.5-35B-A3B-NVFP4-DGX-Spark) |

¹ Nemotron's `~93 tok/s` is the L.A.I.L lab bench (`Decode (c=1, median)`, run id `20260812T131332Z_d0f7ae`), not the prose harness used by the 2× Spark recipes.
² The Ornith README publishes `vllm bench serve` output throughput per command and says not to treat any one tok/s as a general decode number; its max-concurrency 1 run reported peak concurrent 2.

"Last verified" is the date stated in the README (Nemotron run id, Ornith "19 Aug 2026"), the re-bench date for GLM and DeepSeek (`evidence/rebench-20260902T*` in each repo), otherwise the last commit on `main`.

## Methodology

- **Decode bench (2× Spark recipes).** `bench_decode.py` is frozen in each repo. Decode only, streamed greedy, thinking off, 200 completion tokens, 3-run median, concurrency 1 and 2 (`python3 bench_decode.py` runs both phases at c=1,2). Only prose decode (free text) is published in this table. Structured stays in each recipe's `bench.txt` evidence. Do not copy it into a prose cell. Tables report median per-stream decode tok/s, aggregate tok/s, and TTFT p50, run against the repo's default `run.sh` settings.
- **Decode bench (1× Spark recipes).** Ornith uses `vllm bench serve` inside the live container, one bench at a time, each table labelled with the `max-concurrency` that was passed. Nemotron numbers come from the L.A.I.L lab bench with the run id stated in its README.
- **One change at a time.** Knobs (spec tokens, KV pin, batched tokens, CUDA graph ladder, async scheduling, MoE backend) are changed one at a time against the frozen table and reverted when they do not beat noise or when they regress another cell. The READMEs record the reverted values and the numbers that drove the revert.
- **Refuse-guards.** `run.sh` exits instead of booting a setting that was measured to fail: GLM refuses `--max-model-len` above 327,680 on fp8 KV and any MoE backend other than marlin; DeepSeek refuses `--max-model-len` above 1,048,576, a KV pin below 12 GiB when the window is above 289,024, `MAX_NUM_SEQS` above 2, and `NUM_SPECULATIVE_TOKENS` not divisible by 3; Qwen3.8 NVFP4 refuses `--max-model-len` above 1,048,576, `MAX_NUM_SEQS` above 8, global `marlin` (unquantized MTP rejects it), and `flashinfer_cutlass`. `FORCE_UNSAFE_*=1` overrides.
- **Agent-readiness probes.** Unique-salt needle prefill (`needle_probe.py`, pass/fail for a context window, not a tok/s bench; `--concurrency 2` also checks the serve survives two long prompts), thinking-off smoke (`temperature 0`, content must not start with chain-of-thought), tool-call smoke (`smoke_tools.py`), vision smoke (`smoke_vision.py`, must not return HTTP 400 `is not a multimodal model`), and a Hermes-style tool follow-up on GLM. Ornith's ready-gate is a `pong` chat with both `thinking:false` and `enable_thinking:false`.
- **Evidence.** Hypotheses, the decision log, the trail, probe outputs, and `bench_decode.py` `SUMMARY` JSON are published in each repo's `evidence/` directory.
- **Memory.** Unified memory is read with `free -h`, not `nvidia-smi`.

## Hardware

- 2× recipes: "Two DGX Sparks on the QSFP RoCE link (stock `10.100.8.1` / `10.100.8.2`)", tensor-parallel 2, head `spark1` and worker `spark2`. Pin `NCCL_IB_HCA`: GB10 exposes four HCAs and two of them are DOWN; unpinned NCCL picks a dead one and fails with `unhandled system error`. This lab uses `enp1s0f1np1` / `rocep1s0f1`; some Spark cookbooks use `enp1s0f0np0` / `rocep1s0f0`.
- 1× recipes: one NVIDIA DGX Spark GB10 (`sm_121`), 121Gi unified memory total per `free -h`.
- All numbers were measured on `spark1` (and `spark2` for TP=2) in the L.A.I.L lab.

## Who this is for

Engineers with one or two DGX Sparks who want a serve command that has already been booted and measured on the same hardware, with the settings that fail encoded as refusals in `run.sh` rather than as footnotes.

## Related

- [L.A.I.L](https://github.com/sfxnz/L.A.I.L): the serve and eval console that produced the Nemotron numbers. Not a recipe.
