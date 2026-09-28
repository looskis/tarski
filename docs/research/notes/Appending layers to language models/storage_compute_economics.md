# Storage, Memory, and Compute Economics: Added Modules on a Frozen Backbone vs. Per-Task Full Fine-Tuning

Research date: 2026-09-27. All pricing was fetched on this date unless another date is given. Paper dates are arXiv submission or revision dates.

---

## 1. Storage: bytes per task, and how it scales to 10s–1000s of tasks or tenants

### Takeaway
Per-task storage drops by roughly 3 to 6 orders of magnitude when you keep one frozen backbone plus small per-task modules:
- A full 175B fine-tuned copy is about 350 GB. A LoRA adapter for it is about 35 MB.
- A full 7B copy is about 14.5 GB. A LoRA adapter for it is about 14 MB.
- A soft prompt or linear head is only kilobytes.

At 100 to 1,000 tasks, the backbone becomes the dominant storage cost, and total storage grows only slowly with the number of tasks. N full copies grow linearly (N × model size).

### Cited Findings
- **GPT-3 175B with LoRA:** with r=4 on the query/value projections only, the checkpoint shrinks about 10,000× (350 GB to 35 MB). Storing 100 adapted models takes 350 GB + 35 MB × 100 ≈ 354 GB, versus 100 × 350 GB ≈ 35 TB for 100 full fine-tunes (LoRA paper, Hu et al., Jun 2021). — [LoRA (arXiv 2106.09685, ar5iv)](https://ar5iv.labs.arxiv.org/html/2106.09685)
- **7B adapter size:** the LoRA adapter predibase/magicoder is 13.6 MB. mistralai/Mistral-7B-v0.1 is 14.48 GB, so the adapter is under 1/1000th of the base. Loading 30 adapters caused only a 3% increase in VRAM (Hugging Face TGI Multi-LoRA blog, Jul 2024). — [HF blog: TGI Multi-LoRA](https://huggingface.co/blog/multi-lora-serving)
- **3B adapter size:** a PEFT LoRA checkpoint for bigscience/T0_3B is 19 MB, versus 11 GB for the full model. A Stable Diffusion LoRA checkpoint is 8.8 MB (Hugging Face PEFT README). — [huggingface/peft](https://github.com/huggingface/peft)
- **Houlsby adapters:** they get within 0.4% of full fine-tuning on GLUE while adding only 3.6% parameters per task (Houlsby et al., Feb 2019). — [arXiv 1902.00751](https://arxiv.org/abs/1902.00751)
- **Prefix-tuning:** it learns about 0.1% of the parameters and matches fine-tuning in the full-data setting on table-to-text tasks (Li & Liang, Jan 2021). — [arXiv 2101.00190](https://arxiv.org/abs/2101.00190)
- **Prompt tuning:** for T5-XXL, each fully tuned copy needs 11B parameters. A tuned 5-token prompt needs only 20,480 parameters per task, over five orders of magnitude fewer. One frozen model can serve mixed-task batches (Lester et al., Apr 2021). — [arXiv 2104.08691](https://arxiv.org/abs/2104.08691)
- **Hidden-state probes:** probes on frozen Llama-3.2-3B hidden states range from about 0.003M parameters (direct pooling) and about 0.10M (scoring-attention gate) to about 35M (multi-head self-attention) (arXiv 2601.13288v2, Apr 2026). — [BERTology View of LLM Orchestrations](https://arxiv.org/html/2601.13288)
- **Moderation probe:** a streaming-moderation MLP probe on layer 21 of the generating LLM adds about 8 MB of weights. A separate 8B guard model adds 15.5 GB (arXiv 2606.10487, Jun 2026). — [Stop Early, Spend Less](https://arxiv.org/html/2606.10487)

### Inferences
- **Scaling a 7B model to 1,000 tenants** (bf16, sizes from the HF blog):
  - Full fine-tunes: 1,000 × 14.48 GB ≈ 14.5 TB.
  - One base plus 1,000 LoRAs: 14.48 GB + 1,000 × 13.6 MB ≈ 28 GB, about 500× less.
  - At 10 tenants: 145 GB versus about 14.6 GB.
- **Linear heads are smaller still.** A head on a 4,096-d hidden state with 10 classes has about 41K parameters, about 80 KB in bf16, so 1,000 heads are about 80 MB. This is simple arithmetic, not a measured result.
- **Adapter size depends on rank and target modules.** It is not fixed. The 35 MB GPT-3 figure used r=4 on Q/V only. Applying LoRA to all linear layers (including MLPs), as "LoRA Without Regret" recommends for quality, makes adapters several times larger. They are still orders of magnitude smaller than full copies.

### Gaps
- I found no production-scale published audit (e.g., from a SaaS vendor) of actual adapter storage bills for thousands of tenants. The scaling above is arithmetic from per-adapter sizes.
- I did not extract the efficiency tables from "Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey" (Han et al., 2024, arXiv 2403.14608). The numbers here come from the primary papers instead.

---

## 2. Training memory and compute: who still backpropagates through the frozen backbone?

### Takeaway
Methods that insert trainable parameters inside the network still need a full forward pass plus backpropagation of activation gradients through every layer above the lowest adapted layer. This covers LoRA, mid-network adapters, and prefix/prompt tuning.
- They skip weight gradients and optimizer states for the frozen weights. That cuts memory sharply and cuts compute to about 2/3 of full fine-tuning per pass.
- Activation memory remains the bottleneck, so LoRA and adapters save only about 26% of training memory on T5-base.

Methods that never backpropagate through the backbone save much more. These are top-only heads, side networks such as Ladder Side-Tuning, and training on cached activations.
- LST saves about 69% of training memory.
- Cached-feature heads reduce the backbone cost to a single forward pass.

### Cited Findings
- **LoRA on GPT-3 175B:**
  - Training VRAM drops from 1.2 TB to 350 GB.
  - Training is about 25% faster than full fine-tuning, because gradients are not computed for most parameters (Hu et al., 2021).
  - Source: [LoRA (ar5iv)](https://ar5iv.labs.arxiv.org/html/2106.09685)
- **LoRA compute per pass:** slightly more than 2/3 of the FLOPs of full fine-tuning, about 2N² + 6NR multiply-adds versus 3N² per layer-matrix pass. Training memory is "only slightly larger" than inference. The optimal LoRA learning rate is about 10× the full fine-tuning LR (Thinking Machines "LoRA Without Regret", Sep 29, 2025). — [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/)
- **QLoRA memory:**
  - 16-bit full fine-tuning of a 65B model needs more than 780 GB of GPU memory. QLoRA fits it in under 48 GB (one GPU).
  - Guanaco-65B trained in 24 hours on one GPU, and Guanaco-33B in under 12 hours on a consumer GPU.
  - For LLaMA-7B at batch size 1, the 4-bit base uses 5,048 MB, the LoRA input gradients 567 MB, and the LoRA parameters only 26 MB (Dettmers et al., May 2023).
  - Source: [QLoRA (ar5iv 2305.14314)](https://ar5iv.labs.arxiv.org/html/2305.14314)
- **Measured PEFT memory on A100 80GB** (Hugging Face PEFT README):

  | Model | Full fine-tuning | PEFT-LoRA | PEFT-LoRA + DeepSpeed CPU offload |
  |---|---|---|---|
  | T0_3B | 47.14 GB GPU | 14.4 GB GPU | 9.8 GB GPU |
  | bloomz-7b1 | OOM | 32 GB GPU | 18.1 GB GPU |
  | mt0-xxl (12B) | OOM | 56 GB GPU | 22 GB GPU |

  Source: [huggingface/peft](https://github.com/huggingface/peft)
- **Ladder Side-Tuning (LST):**
  - It saves 69% of full fine-tuning memory, versus about 26% for Adapters and LoRA, which it describes as 2.7× more memory savings.
  - On T5-base: 5.5 GB for LST, 17.6 GB for full fine-tuning, and 12.6–13.0 GB for Adapters/LoRA.
  - It updates 1.74% of parameters on T5-base and 0.08% on T5-3B. GLUE average is 84.1 for LST versus 85.2 for full fine-tuning and 85.3 for Adapter/LoRA.
  - It avoids "expensive backpropagation of a large backbone network" by feeding intermediate activations into a separate side network through shortcut connections.
  - Trained on one A6000 48 GB (Sung et al., Jun 2022).
  - Source: [LST (ar5iv 2206.06522)](https://ar5iv.labs.arxiv.org/html/2206.06522)
- **Mixed-precision Adam memory:** it needs about 16 bytes per parameter for fp16 params, fp16 grads, and fp32 master params plus two moments, before activations. A 1.5B-parameter GPT-2 therefore needs at least 24 GB (ZeRO paper, Rajbhandari et al., Oct 2019). — [arXiv 1910.02054](https://arxiv.org/abs/1910.02054)
- **Per-adapter training cost:**
  - LoRA Land trained each of 310 adapters on a single A10G (24 GB) with 4-bit quantization, rank 8, 40,000 steps, and batch size 1. [LoRA Land (arXiv 2405.00732)](https://arxiv.org/html/2405.00732v1)
  - Predibase reported its 25 LoRA Land Mistral-7B adapters cost under $8.00 each on average to fine-tune (Feb 2024). The page is now branded Rubrik. [Predibase blog](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4); the ~$8 figure is also repeated in the [HF TGI blog](https://huggingface.co/blog/multi-lora-serving)

### Inferences
- **Standard FLOP accounting:** training costs about 6N FLOPs per token (2N forward + 4N backward). LoRA and adapters skip the weight-gradient half of the backward pass, giving about 4N per token (≈2/3), which matches Thinking Machines' figure.
- **Top-only heads on cached features** pay about 2N per token once to extract features. After that:
  - K heads × E epochs cost almost nothing.
  - Compare K × E × 6N for K separate full fine-tunes.
  - Example: K=10 tasks, E=3 epochs gives about 180N versus about 2N per token of data, roughly 90× less backbone compute.
- **Top-only and side methods also shrink activation memory.** Activations for backprop through the backbone are not stored, which is exactly what LST measures.
- **Memory ordering for a 7B model** (from the numbers above):
  - Full fine-tuning (≥112 GB + activations): usually does not fit on one 80 GB GPU.
  - LoRA in bf16 (~32 GB on A100).
  - QLoRA (~5–6 GB base + small overhead for 7B).
  - Cached-feature head training (only the head's parameters and a batch of cached vectors on GPU; the backbone need not be resident).
- **Prefix/prompt tuning still backpropagates through all layers,** since the soft tokens enter at the input or at every layer. Its training memory is therefore LoRA-like, not LST-like, even though its storage per task is tiny.

### Gaps
- I did not find a precise, sourced number for QLoRA's training slowdown from dequantization relative to bf16 LoRA. It is commonly reported as slower, but I did not verify a figure.
- I did not find head-to-head wall-clock or FLOP measurements that compare LoRA, prompt tuning, and cached-feature heads under identical conditions.

---

## 3. Feature caching: compute the backbone forward once, train many heads cheaply

### Takeaway
Recent (2026) papers explicitly pre-extract and cache frozen-LLM hidden states and then train tiny probes on them. They report probe sizes from about 3K to 35M parameters, near-zero added inference compute, and one to four orders of magnitude less compute than separate guard or classifier models. The cost is some accuracy on harder tasks.

### Cited Findings
- **Token- and layer-selective probes (arXiv 2601.13288v2, Apr 27, 2026):**
  - Hidden states were "pre-extracted and cached from frozen LLMs before probe training". This decouples backbone inference from classifier training and allows larger batches.
  - The probes classify during the same forward pass used for generation.
  - Backbone: Llama-3.2-3B-Instruct, with validation on GPT-OSS-20B and Qwen3-30B-A3B.
  - The 35M-parameter MHA probe reached 84.51 F1 on ToxicChat, versus 82.2 for the standalone ToxicChat-T5-large.
  - Latency: the probe adds about 13.8 ms per sample on a 26.4 ms base. Guard-then-serve pipelines take 88.3 ms (ToxicChat-T5, 780M) to 123.2 ms (Llama Guard 3).
  - Peak GPU memory: about 6.9 GB with probes versus 22.8 GB for the largest guard pipeline.
  - Source: [arXiv 2601.13288](https://arxiv.org/html/2601.13288)
- **Streaming moderation probe (arXiv 2606.10487, Jun 9, 2026):** an MLP-token head on layer 21 of a frozen LLM, compared on BeaverTails (human labels):

  | Metric | Probe | Post-hoc 8B guard |
  |---|---|---|
  | Added memory | 8 MB | 15.5 GB |
  | Compute per request | 0.6 GFLOP | 8.02 TFLOP |
  | Latency p50/p95/p99 | 0.1/0.1/0.1 ms | 448/551/602 ms |
  | F1 | 0.795 (calibrated) | 0.855 |

  The authors describe this as "two-to-four orders of magnitude less moderation compute." — [arXiv 2606.10487](https://arxiv.org/html/2606.10487)
- **Probing repository:** a public repo describes caching per-layer hidden states to disk in one LLM pass. Many probe variants can then be swept "without re-running the LLM". — [gmeyoyan/LLM-Probing-Classifiers](https://github.com/gmeyoyan/LLM-Probing-Classifiers) (seen in search results; I did not review it in depth)
- **LST as partial caching:** LST's side network consumes backbone intermediate activations, and the backbone gets no backward pass. Because the backbone is frozen, these activations could in principle be cached across epochs (LST paper, Jun 2022). — [LST (ar5iv)](https://ar5iv.labs.arxiv.org/html/2206.06522)

### Inferences
- **Caching only works if everything below the head stays frozen.** No LoRA or adapter can sit below the tapped layer, and inputs must be static. Any trainable module inside the backbone invalidates the cache.
- **Cache storage is a real cost.** Caching a 4,096-d bf16 vector per example is 8 KB/example (1M examples ≈ 8 GB). Caching per-token, all-layer states, as the token- and layer-selective probes need, is far larger: tokens × layers × d × 2 bytes. This is arithmetic, not from a source.
- **Serving cost:** one forward pass on a shared backbone feeds many heads. The marginal cost per extra head is about head FLOPs, e.g., 0.6 GFLOP versus about 8 TFLOP for a separate 8B guard. This is the cleanest "multi-head on one shared encoder" economics found.

### Gaps
- I found no paper that quantifies "N heads trained on one cached-feature store" with an explicit cost curve over N (e.g., cost per additional task). The savings are implied by the probe papers, not directly measured across many tasks.
- I found no production-system writeup (e.g., from a large company) with dollar numbers for shared-encoder multi-head serving.

---

## 4. Serving: multi-adapter / multi-tenant systems vs. N separate fine-tuned models

### Takeaway
Multi-LoRA serving systems keep one base model in GPU memory and batch requests across different adapters.
- They serve from dozens (LoRAX, TGI) to 1,000–2,000+ (S-LoRA, compressed LoRAs) adapters per GPU.
- Throughput is 4× to 12× (up to 57.9× against naive baselines) that of serving merged separate models or naive PEFT.
- Added per-token latency is small: about 2 ms per token for Punica, and about 7 ms between 1 and 25 adapters for LoRAX.

The marginal cost per additional tenant adapter is near zero, while N dedicated deployments cost about N × GPU.

### Cited Findings
- **S-LoRA (Sheng et al., Nov 2023; v3 Jun 2024):**
  - Serves 2,000 LoRA adapters simultaneously on a single GPU, bounded by host memory.
  - Up to 4× throughput versus vLLM-packed, and up to 30× versus HuggingFace PEFT.
  - Tested on A10G 24GB and A100 40/80GB with Llama-7B/13B/30B/70B, at ranks 8–64.
  - Uses Unified Paging, a single memory pool for adapter weights and KV cache, plus custom heterogeneous-batching CUDA kernels.
  - The merged-weight approach's "performance declines with more than 2 adapters, primarily because of the time-consuming switch between adapters."
  - Sources: [S-LoRA (ar5iv 2311.03285)](https://ar5iv.labs.arxiv.org/html/2311.03285); [arXiv abs](https://arxiv.org/abs/2311.03285)
- **Punica (Chen et al., Oct 2023; MLSys 2024):** its SGMV kernel batches different LoRA models while holding a single copy of the base model. It reaches 12× higher throughput on a fixed-size GPU cluster than state-of-the-art LLM serving systems, adding only 2 ms latency per token. — [arXiv 2310.18547](https://arxiv.org/abs/2310.18547); [MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html)
- **dLoRA (Wu et al., OSDI 2024, Jul 2024):**
  - Dynamically merges and unmerges adapters, and co-migrates requests and adapters across replicas.
  - Up to 57.9× throughput versus vLLM and 26.0× versus HF PEFT.
  - Up to 1.8× lower average latency than S-LoRA.
  - Source: [USENIX OSDI '24](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
- **LoRAX (LoRA Land report, Apr/May 2024):**
  - 25 LoRA fine-tuned Mistral-7B models served on a single A100 80GB.
  - With 50 concurrent users across 25 adapters: average TTFT 153.89 ms and total request time 3,336 ms.
  - TTFT at or under about 200 ms.
  - Only a 7.21 ms difference between 1 adapter and 25 adapters under load, at 12–13.5 ms per token.
  - Source: [LoRA Land (arXiv 2405.00732)](https://arxiv.org/html/2405.00732v1)
- **Hugging Face TGI Multi-LoRA (Jul 2024):**
  - 30 adapters loaded for a 3% VRAM increase.
  - The example deployment reached 75 requests/s on one nvidia-l4 (avg 450 input / 234 output tokens), at $0.8/hour for Mistral-7B on L4.
  - "Even when we add many more models with TGI Multi-LoRA the cost is the same per token."
  - All adapters must share the same base model.
  - Source: [HF blog](https://huggingface.co/blog/multi-lora-serving)
- **Compress then Serve (Jun 2024; rev. May 2025):** joint compression of LoRAs into a shared basis serves up to 1,000 LoRAs while keeping 80% of single-LoRA throughput. — [arXiv 2407.00066](https://arxiv.org/abs/2407.00066)
- **vLLM:** per a 2025 multi-LoRA serving paper (search-result summary; I did not fetch it), vLLM implements S-LoRA-style multi-LoRA memory paging and Punica-derived Triton kernels for heterogeneous LoRA batching. The same source says throughput dips slightly as adapter count grows, then flattens (around 100 adapters in its experiments), because the number of adapters active per batch stays bounded. — [arXiv 2505.03756](https://arxiv.org/html/2505.03756v1)
- **AWS SageMaker LMI (May 21, 2024):** supports unmerged multi-LoRA batches on vLLM/LMI-Dist backends. Adapters are tiered across GPU, CPU (`option.max_cpu_loras`), and disk. The post gives no benchmark numbers. — [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/efficient-and-cost-effective-multi-tenant-lora-serving-with-amazon-sagemaker/)
- **Multi-head on a shared backbone:** probes classify during the same forward pass as generation, which removes a separate guard or classifier call. Latency is 13.8 ms added versus 88–123 ms for guard pipelines, and peak memory is 6.9 GB versus 22.8 GB. — [arXiv 2601.13288](https://arxiv.org/html/2601.13288)

### Inferences
- **Serving N separate fine-tuned 7B models needs about N × 14.5 GB just for weights.**
  - One A100 80GB can hold roughly 4–5 such copies, with little room left for KV cache.
  - With multi-LoRA, one GPU serves 25 (LoRAX demo) to 2,000 (S-LoRA) adapters.
  - The GPU-count ratio therefore scales with N once N exceeds a handful.
- **Baseline caveat:** the large multipliers (S-LoRA 30×, dLoRA 57.9×) are against baselines without native multi-LoRA: HF PEFT, older vLLM, or "vLLM-packed" (separate merged copies). Today's vLLM, TGI, and LoRAX all include multi-LoRA batching. The residual gap between a multi-LoRA system and a single-model deployment is the small overhead (about 2–7 ms/token or a few percent VRAM), not 4–57×.
- **The "vLLM-packed" baseline in S-LoRA is essentially the "N separately deployed fine-tuned models" scenario.** The 4× figure is thus the closest published head-to-head for "adapters vs. N full copies" on the same hardware. This is based on my understanding of the S-LoRA setup; I did not verify its exact baseline definition in this pass.

### Gaps
- I did not retrieve official vLLM documentation benchmarks (e.g., max_loras/max_lora_rank throughput curves) or current 2026 vLLM multi-LoRA numbers.
- There is no published dollar-per-tenant comparison of multi-LoRA serving versus dedicated endpoints beyond TGI's qualitative "same cost per token" claim.
- A DEV Community post reports running 40 adapters on 2×A100 with vLLM and about 60 ms p50 grouped-GEMM overhead. It is a low-authority blog I did not fetch, so treat it as anecdotal. — [DEV Community post](https://dev.to/marcuswwchen/serving-40-lora-adapters-on-one-base-model-the-throughput-we-got-m2n)

---

## 5. Inference overhead of added layers: sequential adapters, merged LoRA, added depth, heterogeneous batching

### Takeaway
- **Sequential adapters add real latency at small batch sizes:** about 20–30% at batch 1 on GPT-2 medium, falling to about 2–3% at large batches.
- **LoRA can be merged for zero overhead,** but only when all requests in a batch use the same adapter.
- **Unmerged multi-LoRA adds a small per-token cost.** That cost is memory-bound during decode and becomes significant at hundreds of adapters (PreFT shows about 1.6–2.4× throughput left on the table).
- **Added depth (block expansion) adds compute roughly in proportion to the added parameters.**
- **Side networks and top heads can run in parallel** or add negligible cost.

### Cited Findings
- **Adapter latency (LoRA paper Table 1):** GPT-2 medium on an NVIDIA Quadro RTX8000, compared against the fine-tuned/LoRA-merged baseline.

  | Batch, sequence length | Baseline | AdapterL | AdapterH |
  |---|---|---|---|
  | Batch 1, seq 128 | 19.8 ms | +20.7% | +30.3% |
  | Batch 16, seq 256 | 338.0 ms | +5.0% | +8.4% |
  | Batch 32, seq 512 | 1449.4 ms | +2.2% | +3.0% |

  Source: [LoRA (ar5iv)](https://ar5iv.labs.arxiv.org/html/2106.09685)
- **The merge trade-off:** "it is not straightforward to batch inputs to different tasks with different A and B in a single forward pass, if one chooses to absorb A and B into W to eliminate additional inference latency" (LoRA paper). — [LoRA (ar5iv)](https://ar5iv.labs.arxiv.org/html/2106.09685)
- **Merged-weight serving degrades** with more than 2 adapters, because of adapter switching time (S-LoRA). — [S-LoRA (ar5iv)](https://ar5iv.labs.arxiv.org/html/2311.03285)
- **Measured overhead of unmerged batching:** Punica adds about 2 ms/token. [Punica](https://arxiv.org/abs/2310.18547) LoRAX shows a 7.21 ms difference between 1 and 25 adapters under load. [LoRA Land](https://arxiv.org/html/2405.00732v1)
- **PreFT (May 14, 2026):**
  - Decode is memory-bound, and LoRA's down-projection has poor arithmetic intensity on H100 FP16.
  - Restricting adapters to prefill tokens only increased vLLM multi-LoRA throughput by 1.87–1.9× (Llama 3.1 70B, 512 adapters), 1.83–2.21× (Llama 3.1 8B), and 1.63–2.39× (Qwen 2.5 0.5B) on H100 80GB.
  - Accuracy was at parity on IFEval/MMLU/GSM8K for SFT, with a slight RL lag on GSM8K (−2.04 points).
  - Source: [arXiv 2605.14217](https://arxiv.org/html/2605.14217)
- **LST at inference:** almost the same inference memory as full fine-tuning (0.88 GB vs 0.86 GB). Side-network layers can be "computed in parallel". — [LST (ar5iv)](https://ar5iv.labs.arxiv.org/html/2206.06522)
- **Block expansion:** LLaMA Pro expands LLaMA2-7B into LLaMA Pro-8.3B by inserting Transformer blocks and tuning only those blocks on a new corpus (ACL 2024; Jan 2024). — [arXiv 2401.02415](https://arxiv.org/abs/2401.02415)
- **Probe overhead depends on probe size.** The 35M MHA probe adds about 13.8 ms on a 26.4 ms 3B base (about 50%) [arXiv 2601.13288](https://arxiv.org/html/2601.13288). An 8 MB MLP probe adds about 0.1 ms [arXiv 2606.10487](https://arxiv.org/html/2606.10487).

### Inferences
- **Block expansion from about 6.7B to 8.3B parameters** implies roughly +20–25% per-token FLOPs and weight-memory bandwidth at inference, for every request, permanently. Unlike LoRA, this depth cannot be merged away. This is arithmetic from the parameter counts; the exact number of added blocks was not confirmed in the fetched abstract.
- **Latency ordering by how "free" the added module is at inference:**
  1. Merged LoRA (0, but single-task per batch).
  2. Prefix/prompt tuning (extra KV entries or tokens only).
  3. Top head or probe (≈0 to small).
  4. Unmerged multi-LoRA (a few ms/token, and memory-bound decode).
  5. Parallel side network (extra FLOPs, but parallelizable).
  6. Sequential bottleneck adapters (+2–30% depending on batch).
  7. Block expansion (+~20–25% compute, always).

### Gaps
- The AdapterDrop paper (arXiv 2010.11918) reportedly quantifies adapter training speedups and inference slowdowns, but the abstract page I fetched contained no numbers, so I do not cite a specific percentage.
- I did not find a direct latency measurement of LLaMA Pro versus LLaMA2-7B.

---

## 6. Where the savings disappear or trade-offs appear

### Takeaway
The efficiency case weakens in five situations:
1. **Hard or broad tasks:** code, math, continued pretraining, and large datasets, where LoRA "learns less" than full fine-tuning.
2. **Larger batch sizes** in LoRA training.
3. **Tiny probes,** which lose some accuracy versus dedicated guard models.
4. **Base-model upgrades.** Every adapter must be retrained, e.g., about 4 A100-hours per adapter, or about 8,000 GPU-hours for 2,000 adapters.
5. **Low volume.** Fixed GPU cost exceeds pay-per-token API cost.

Hosted-API fine-tuning also carries a per-token inference markup (about 2× base price at OpenAI for 4.1-nano/4o-mini).

### Cited Findings
- **"LoRA Learns Less and Forgets Less" (Biderman et al., May 2024; TMLR):**
  - In standard low-rank settings, "LoRA substantially underperforms full finetuning" on programming and math, both for instruction fine-tuning (~100K pairs) and continued pretraining (20B tokens).
  - Full fine-tuning learns perturbations of 10–100× higher rank than typical LoRA configurations.
  - LoRA better preserves out-of-domain performance.
  - Source: [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- **"LoRA Without Regret" (Sep 29, 2025):**
  - LoRA matches full fine-tuning on small-to-medium SFT/reasoning datasets and in RL (even rank-1) when applied to all weight matrices.
  - It underperforms when the dataset exceeds LoRA capacity, at large batch sizes (a persistent gap independent of rank), and with attention-only LoRA.
  - Source: [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- **LoRA Land (Apr/May 2024):** 224 of 310 LoRA models beat GPT-4, averaging +9.5 points over GPT-4 and +25 over base. GPT-4 won on broader or complex tasks such as Python coding (MagicCoder) and MMLU. The abstract reports 6 of 31 tasks. — [arXiv 2405.00732](https://arxiv.org/html/2405.00732v1); [HF paper page](https://huggingface.co/papers/2405.00732)
- **Probe accuracy gap:** the 8 MB streaming probe reached F1 0.795 versus 0.855 for an 8B post-hoc guard, trading about 6 F1 points for more than 10,000× less compute. — [arXiv 2606.10487](https://arxiv.org/html/2606.10487)
- **Side-network gap:** LST averaged 84.1 on GLUE versus 85.2 for full fine-tuning and 85.3 for Adapter/LoRA. It buys memory at about 1 point of accuracy. — [LST](https://ar5iv.labs.arxiv.org/html/2206.06522)
- **Base-model upgrades (ReLoRA, May 23, 2026):**
  - When base LLMs update, deployed LoRAs become incompatible.
  - "Fine-tuning a single LoRA instance on a moderately sized dataset takes approximately 4 hours on an NVIDIA A100 GPU. Updating 2,000 LoRA instances would require roughly 8,000 GPU hours."
  - ReLoRA (adaptive initialization plus scheduled regularization) gives up to 8.9× faster time-to-readiness and 2.32× end-to-end rollout speedup versus retraining from scratch, tested on LLaMA2-7B, LLaMA3.1-8B, and Mistral-7B.
  - Source: [arXiv 2606.02606](https://arxiv.org/html/2606.02606)
- **Adapter transfer methods:**
  - LoRA modules are tied to their base model, and retraining after deprecation needs the original data. [Cross-LoRA (arXiv 2508.05232)](https://arxiv.org/pdf/2508.05232)
  - Trans-LoRA (NeurIPS 2024) proposes nearly data-free transfer via synthetic data. [arXiv 2405.17258](https://arxiv.org/abs/2405.17258)
  - Cross-LoRA proposes data-free, training-free transfer across heterogeneous LLMs. [arXiv 2508.05232](https://arxiv.org/pdf/2508.05232)
- **Hosted fine-tuning prices (OpenAI pricing page, fetched 2026-09-27):**
  - Training per 1M tokens: gpt-4.1 $25, gpt-4.1-mini $5, gpt-4.1-nano $1.50, gpt-4o $25, gpt-4o-mini $3. o4-mini is $100/hour.
  - Fine-tuned model inference per 1M input/output: gpt-4.1-nano $0.20/$0.80, gpt-4o-mini $0.30/$1.20, gpt-4.1-mini $0.80/$3.20, o4-mini $4/$16.
  - Standard base prices: gpt-4.1-nano $0.10/$0.40, gpt-4o-mini $0.15/$0.60.
  - Source: [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
  - Conflict: an aggregator claimed GPT-4.1 trains at about $3/M tokens and fine-tuned GPT-4o inference is 1.5× base ([pricepertoken](https://pricepertoken.com/fine-tuning), [aicostcheck](https://aicostcheck.com/blog/ai-fine-tuning-costs-2026)). The official page lists $25/M for gpt-4.1 training, so defer to the official page.
- **Anthropic (fetched 2026-09-27):** the Claude API pricing page lists no fine-tuning product or price. — [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)

### Inferences
- **Fine-tuned inference markup:** the official OpenAI figures show fine-tuned inference at about 2× the base model's per-token price for gpt-4.1-nano and gpt-4o-mini. So "fine-tune a hosted model" does not save per-message cost; it buys accuracy or shorter prompts.
- **Hosted fine-tunes are pinned to a dated snapshot** (e.g., gpt-4o-mini-2024-07-18). Snapshot deprecation forces retraining, the hosted analogue of adapter non-transferability.
- **Base-model upgrade cost scales linearly with the number of adapters.** At the ReLoRA numbers (4 A100-hours per adapter) and a ballpark A100 rental price, 2,000 adapters cost several thousand to over ten thousand dollars per base-model upgrade. That is still far less than full fine-tuning 2,000 models, but it is a recurring cost that heads-on-cached-features share: the cache must be recomputed and heads retrained whenever the backbone changes. The dollar figure is inferred, not sourced.
- **Low-volume break-even:** a dedicated L4 at $0.8/hour is about $580/month regardless of traffic. See Question 7 for a comparison with API pricing at low volume.

### Gaps
- I found no systematic study that quantifies how often production teams must retrain adapters because of base-model churn, or the total cost of ownership over a model's lifetime.
- I did not verify current (2026) Google Vertex AI or AWS Bedrock fine-tuning prices, e.g., Bedrock custom-model hosting (Provisioned Throughput) or Vertex supervised tuning.

---

## 7. Practical cost comparison for a classification/routing workload

### Takeaway
For fixed-label classification, fine-tuned small encoders cost about $5–$33 per 1M requests on Cloud Run CPU (Jan 2026 prices), with p50 latency around 100–600 ms. Prompting GPT-4o or Claude Sonnet 4.5 cost about $190–$2,700 per 1M requests, and encoders matched or beat LLM accuracy on 3 of 4 benchmarks.

At September 2026 prices, the cheapest hosted LLMs narrow the gap, but encoders or LoRA on a shared self-hosted backbone remain one to two orders of magnitude cheaper at high volume. A multi-LoRA 7B server is a middle option: fixed about $0.8/hour per L4, hosting many task adapters at no extra cost.

### Cited Findings
- **"Cost-Aware Model Selection for Text Classification" (arXiv 2602.06370, Feb 6, 2026; pricing snapshot Jan 22, 2026):** BERT, RoBERTa, and DistilBERT fine-tuned (trained on A100, served on Google Cloud Run, 2 vCPU / 2 GiB) versus GPT-4o ($2.50/$10) and Claude Sonnet 4.5 ($3/$15) zero- and few-shot.

  | Dataset | Encoder cost per 1M req | LLM cost per 1M req | Best F1 (encoder vs LLM) | p50 latency (encoder vs LLM) |
  |---|---|---|---|---|
  | SST-2 | $5.19–$7.79 | $192.48–$599.67 | 94.42 vs 94.41 | 97–147 ms vs 326–1,394 ms |
  | AG News | $5.73–$10.44 | $276.00–$1,271.58 | 94.63 vs 91.35 | 108–197 ms vs 332–1,435 ms |
  | DBPedia | $7.03–$10.90 | $463.20–$2,701.89 | 99.40 vs 98.83 | 132–205 ms vs 406–1,127 ms |
  | IMDB | $12.44–$32.98 | $842.78–$2,120.01 | 94.84 vs 96.48 | encoder p50 234–622 ms |

  - IMDB is the one dataset where the LLM wins, with Claude few-shot.
  - Few-shot prompting raised token cost without proportional accuracy gains.
  - Conclusion: encoders deliver "competitive, and often superior, classification performance while operating at one to two orders of magnitude lower cost and latency."
  - Source: [arXiv 2602.06370](https://arxiv.org/html/2602.06370v1)
- **OpenAI small-model pricing (fetched 2026-09-27), per 1M input/output tokens:** gpt-5-nano $0.05/$0.40, gpt-5-mini $0.25/$2.00, gpt-4.1-nano $0.10/$0.40, gpt-4o-mini $0.15/$0.60. The Batch API is 50% off. — [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
- **Anthropic pricing (fetched 2026-09-27), per 1M input/output tokens:**
  - Claude Haiku 4.5 $1/$5 (batch $0.50/$2.50; cache hits 0.1× input). Claude Sonnet 5 $2/$10.
  - Claude 4.7+ models use a tokenizer producing about 30% more tokens for the same text. Haiku 4.5 uses the previous tokenizer.
  - Anthropic's own example: about $37 per 10,000 support tickets (about 3,700 tokens each) on Haiku 4.5.
  - Source: [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **Google Gemini API pricing (fetched 2026-09-27), per 1M input/output tokens:**
  - Gemini 3.5 Flash-Lite $0.30/$2.50 (batch $0.15/$1.25).
  - Gemini 3.8 Flash $0.75/$3.75 (through Dec 31, 2026).
  - Gemini Embedding 2: $0.20 per 1M text tokens (batch $0.10).
  - Source: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- **Self-hosted multi-LoRA 7B:** Mistral-7B on nvidia-l4 at $0.8/hour sustained 75 req/s (450 input / 234 output tokens) with TGI Multi-LoRA. Adding adapters does not change per-token cost. — [HF TGI blog](https://huggingface.co/blog/multi-lora-serving)
- **LoRA fine-tuned 7B quality:** 4-bit LoRA fine-tuned models outperformed GPT-4 by about 10 points on average across 31 mostly classification/extraction-style tasks, and adapters cost under $8 each to train. — [LoRA Land](https://arxiv.org/abs/2405.00732); [Predibase blog](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4)

### Inferences
Illustrative arithmetic, not a published benchmark. Workload: 1M messages/day, about 300 input tokens (message plus instructions) and about 5 output tokens (label). Prices as of 2026-09-27, before batch and caching discounts:

| Option | Input cost/day | Output cost/day | Total/day | Total/month |
|---|---|---|---|---|
| gpt-5-nano | 300M × $0.05/M = $15 | 5M × $0.40/M = $2 | ≈$17 | ≈$510 |
| Gemini 3.5 Flash-Lite | 300M × $0.30/M = $90 | 5M × $2.50/M ≈ $12.5 | ≈$102 | ≈$3,100 |
| Claude Haiku 4.5 | 300M × $1/M = $300 | 5M × $5/M = $25 | ≈$325 | ≈$9,750 |
| gpt-4o (as in 2602.06370) | 300M × $2.50/M = $750 | 5M × $10/M = $50 | ≈$800 | — |

- **Caveat for reasoning models:** if the chosen model emits reasoning tokens, they are billed as output and can dominate cost. The ≈$17/day figure for gpt-5-nano assumes negligible reasoning output.
- **Fine-tuned encoder:** at the 2602.06370 figures of about $5–$11 per 1M requests on Cloud Run CPU, the workload costs about $5–$11/day (about $150–$330/month).
- **LoRA on a 7B model, self-hosted:**
  - 1M/day averages about 11.6 req/s, well under the 75 req/s one L4 sustained on longer generative requests. One L4 at $0.8/hour costs about $19/day (about $580/month) whether it serves 1 or 30 classification tasks.
  - For K tasks, API cost scales about K× while the multi-LoRA server stays flat until throughput saturates.
- **Embedding API plus local heads:** Gemini Embedding 2 at $0.20/M tokens gives 300M tokens/day ≈ $60/day ($30 batch). Many heads can be trained on the same cached embeddings for free, so this beats per-task LLM prompting once there are multiple labels or tasks per message.
- **Crossover at low volume:**
  - At 10K messages/day, the gpt-5-nano bill is about $0.17/day, versus about $19/day for a dedicated L4. The API wins at low volume.
  - A CPU encoder on serverless Cloud Run scales to zero and stays cheapest.
  - A dedicated GPU (7B LoRA) pays off only at roughly hundreds of thousands to millions of requests/day, or when many tasks share it. At gpt-5-nano prices it is closer to 1M+/day for a single task.
- **The cheapest September 2026 hosted small models are 10–50× cheaper per token than GPT-4o was.** This narrows 2602.06370's "one to two orders of magnitude" gap to roughly 2–3× for gpt-5-nano versus a CPU encoder on this synthetic workload. Encoders still win on tail latency and on-prem/privacy.

### Gaps
- I found no published head-to-head benchmark measuring all three options (small encoder + head, LoRA on 7B, and hosted-LLM prompting) on the same routing workload with 2026 prices and hardware. The comparison above combines separate sources plus arithmetic.
- 2602.06370 used GPT-4o and Claude Sonnet 4.5 (not the cheapest tiers) and CPU serving for encoders (not GPU batching). Encoder costs could be lower with batched GPU inference, and LLM costs lower with nano/Flash-Lite tiers and batch/caching discounts.
- The 2512.19011 paper ("Do You Really Need a GPU to Guard Your LLM? CPU-Class Classifiers…", v3 Jun 2026) appears directly relevant to CPU encoder-head versus GPU guard-model cost, but its PDF could not be parsed, so no numbers are cited. — [arXiv 2512.19011](https://arxiv.org/abs/2512.19011)
