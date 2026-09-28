# Additive Adaptation Architectures for Transformers/LLMs (new layers/blocks/modules on a frozen backbone)

Scope note: covers adapters, block expansion / model growth, side networks, top-only appending vs. depth-wise insertion, expressivity theory, and failure modes. LoRA is mentioned only as a baseline (covered by another researcher). Research conducted 2026-09-27; every finding is dated by paper year (arXiv submission year unless a venue is stated).

## Core question: Can we fine-tune a language model by appending layers rather than adjusting the weights of an existing trained model?

### Takeaway
Yes, and this has been demonstrated repeatedly since 2019: bottleneck adapters inserted into every layer of a frozen BERT get within 0.4 points of full fine-tuning on GLUE with 3.6% new parameters per task (2019). Whole new transformer blocks interleaved into a frozen LLaMA2-7B add large code and math gains with almost no general-domain forgetting (2024). Side networks that never backpropagate through the frozen model get within about 1 point of full fine-tuning at a fraction of the memory (2022–2025). The strongest evidence says *where* the new parameters go matters a great deal. Modules distributed through depth (especially parallel to the FFN) work far better than modules bolted only onto the top, and blocks stacked at the bottom are catastrophic. The main cost is extra inference depth or latency, which is why unmergeable additive modules have lost mindshare to mergeable LoRA-style updates in practice.

### Cited Findings
- **Summary table of representative additive methods (backbone frozen in all cases):**

| Method (year) | Where new params sit | New/trainable params | Reported quality vs full fine-tuning (FT) | Source |
|---|---|---|---|---|
| Houlsby adapters (2019) | 2 bottleneck adapters per layer (after attention projection and after FFN) | 3.6% per task (GLUE); 1.14% per task (17 extra tasks) | GLUE 80.0 vs 80.4 FT (BERT-large) | [Houlsby et al. 2019](https://arxiv.org/abs/1902.00751) |
| Pfeiffer adapter / AdapterFusion (2020) | 1 adapter after FFN per layer; fusion attention over adapters per layer | ~3.6% of BERT-base per adapter (per summary of paper) | 16-task avg: FT 75.51, single adapters 76.05, AdapterFusion 77.33 | [Pfeiffer et al. 2020](https://ar5iv.labs.arxiv.org/html/2005.00247) |
| Compacter (2021) | Low-rank hypercomplex adapters in each layer | 0.047% | "on par" with FT on GLUE; better on SuperGLUE and low-resource | [Mahabadi et al. 2021](https://arxiv.org/abs/2106.04647) |
| MAM / parallel adapter (2021) | Parallel adapter at FFN + prefix at attention | 0.5% (GLUE), 6.7% (XSum/MT) | MNLI/SST2 matches FT (87.4/94.2); XSum R-2 21.90 vs 21.94; WMT16 En-Ro 37.5 vs 37.3 BLEU | [He et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.04366) |
| LLaMA Pro block expansion (2024) | 8 new identity-initialized blocks interleaved into 32 frozen blocks | 8 of 40 blocks trained (7B → 8.3B) | HumanEval 13.05 → 28.66, MBPP 20.09 → 33.20, general tasks ~unchanged | [Wu et al. 2024](https://arxiv.org/html/2401.02415) |
| Ladder Side-Tuning (2022) | Separate thin side network reading downsampled backbone activations | 1.74% | T5-base GLUE 84.1 vs 85.2 FT; memory 5.5 GB vs 17.6 GB | [Sung et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.06522) |
| Quantized Side Tuning (2024) | Side network on 4-bit frozen LLM | small side net | LLaMA-2-70B MMLU 63.9 vs QLoRA 64.1, 56.0 GB vs 95.5 GB | [Zhang et al. 2024](https://arxiv.org/html/2401.07159) |
| Proxy-tuning (2024) | No new layers inside; small tuned model's logit delta added at decode | 7B expert + anti-expert | closes 88% of gap between Llama2-70B base and 70B-chat | [Liu et al. 2024](https://arxiv.org/html/2401.08565) |

- The PEFT literature formally names this family "additive" methods: they "introduce new parameters or layers while freezing original weights," as distinct from selective, reparametrization-based (e.g., LoRA) and hybrid methods. Adapters, soft prompts, Ladder Side-Tuning and (IA)^3 fall under "additive." — [Lialin et al. 2023, "Scaling Down to Scale Up"](https://ar5iv.labs.arxiv.org/html/2303.15647)
- Han et al. (2024) use the same four-way additive / selective / reparameterized / hybrid split in their PEFT survey for large models. — [Han et al. 2024](https://arxiv.org/pdf/2403.14608)
- Pfeiffer, Ruder, Vulić, Ponti (2023) frame adapters as one instance of "modular deep learning." Their taxonomy has parameter composition, input composition and function composition, with routing and aggregation. They argue modularity gives positive transfer, avoids negative interference and aids systematic generalization. — [Pfeiffer et al. 2023, Modular Deep Learning](https://arxiv.org/abs/2302.11529)
- Soft prompt tuning is also additive: it adds virtual tokens rather than layers. It "closes the gap" with and "matches" model tuning once models exceed billions of parameters (T5 up to XXL). Frozen-model prompting also gives better robustness to domain transfer than full model tuning (2021). — [Lester et al. 2021](https://arxiv.org/abs/2104.08691)
- For decoder LLMs (LLaMA-7B/13B, 2023), adapters remain competitive with LoRA. On commonsense reasoning, LLaMA-13B scores Series adapter 79.5, Parallel adapter 81.5 and LoRA 80.5 average, versus 77.0 for zero-shot-CoT ChatGPT. On math reasoning, LLaMA-13B+LoRA averages 65.4 versus 70.4 for GPT-3.5 zero-shot-CoT (EMNLP 2023). — [Hu et al. 2023, LLM-Adapters](https://arxiv.org/html/2304.01933)

### Inferences
- "Append layers, freeze the rest" is a well-established, working paradigm. The design question is not *whether* but *where* (throughout depth versus top-only), *how wired* (sequential, parallel, or side/ladder), and *what the deployment cost is* (latency and memory).
- On NLU, additive methods routinely reach 97–100% of full fine-tuning quality. On harder generation tasks and with very small parameter budgets, a gap persists (see the failure-modes section).
- Additive methods keep the original weights bit-for-bit intact, so the base model's behavior can be recovered exactly by removing the module. That gives a structural guarantee against forgetting which weight-update methods lack, apart from LoRA, which can be unmerged.

### Gaps
- No single head-to-head 2025–2026 benchmark was found that compares all additive families (adapters vs. block expansion vs. side networks) on modern decoder LLMs at a fixed budget. Numbers above come from different backbones, tasks and years, so they are not directly comparable.

## Adapter layers inserted between frozen transformer sublayers (Houlsby, Pfeiffer, AdapterFusion, parallel adapters, Compacter, AdapterDrop)

### Takeaway
Bottleneck adapters inserted into every layer reach within about 0.4 points of full fine-tuning on GLUE with 0.05–4% new parameters per task (2019–2021). Parallel placement beats sequential, and FFN-side placement beats attention-side (He et al. 2021). On generation tasks (XSum summarization, WMT translation), matching full fine-tuning needs a much larger budget (about 6.7% of parameters) than GLUE does (about 0.5%).

### Cited Findings
- **Houlsby adapters (2019, ICML):** small bottleneck modules inserted *twice* per Transformer layer, "after the projection following multi-headed attention and after the two feed-forward layers." Each is a down-projection d→m, a nonlinearity, then an up-projection m→d, with a skip connection so it starts near identity. — [Houlsby et al. 2019 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1902.00751)
- On GLUE the adapters attain "within 0.4% of the performance of full fine-tuning, adding only 3.6%" parameters per task. — [Houlsby et al. 2019 (abstract)](https://arxiv.org/abs/1902.00751)
- With BERT-large, GLUE scores are 80.0 for adapters versus 80.4 for full FT. Total parameters for 9 tasks are 1.3× the base model for adapters versus 9× for fine-tuning. — [Houlsby et al. 2019 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1902.00751)
- On 17 further classification tasks, "variable fine-tuning" of only the top n layers needed 52.9% of parameters per task (9.9× total), while adapters needed 1.14% per task (1.19× total). Adapters also stayed within about 0.4% of full FT on these tasks and on SQuAD. — [Houlsby et al. 2019 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1902.00751)
- Ablation: "removing adapters from the layers 0-4 on MNLI barely affects performance," so higher-layer adapters matter more. — [Houlsby et al. 2019 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1902.00751)
- **Pfeiffer adapter configuration (2020):** a *single* adapter per layer, placed after the FFN sublayer, instead of Houlsby's two. Reduction factors {2, 16, 64} were tested, and 16 gave the best performance/efficiency trade-off. — [Pfeiffer et al. 2020, AdapterFusion (ar5iv)](https://ar5iv.labs.arxiv.org/html/2005.00247)
- **AdapterFusion (2020):** a two-stage method. Stage one trains per-task adapters independently. Stage two learns a per-layer fusion module (attention with Query/Key/Value over the adapter outputs) that "dynamically" mixes them, while backbone and adapters stay frozen. It is "non-destructive," which addresses the catastrophic forgetting and dataset-balancing problems of sequential or multi-task FT. — [Pfeiffer et al. 2020 (abstract)](https://arxiv.org/abs/2005.00247); [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00247)
- AdapterFusion results over 16 NLU tasks: average of full FT 75.51, single-task adapters 76.05, AdapterFusion 77.33. The largest gains are on low-resource tasks: RTE +6.5 and MRPC +5.64 over single adapters. AdapterFusion beat single adapters on 10 of 16 tasks. — [Pfeiffer et al. 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2005.00247)
- **Parallel adapters / unified view (He et al., 2021; ICLR 2022):** this work recasts adapters, prefix tuning and LoRA as modifications of hidden states. They differ in functional form, insertion position, modified representation, and composition function. — [He et al. 2021 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2110.04366)
  - "Parallel adapter is able to beat sequential adapters in all cases."
  - FFN-side modification beats attention-side modification, especially at larger capacity.
  - A scaled parallel adapter improves on the vanilla adapter by 0.56 ROUGE-2.
- XSum (ROUGE-2) scores from He et al. (2021). Full FT 21.94; MAM adapter (6.7%) 21.90; parallel adapter at FFN (12.3%) 21.41; LoRA-FFN (7.2%) 21.29; sequential adapter (7.2%) 20.89; prefix tuning (3.6%) 20.46. — [He et al. 2021 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2110.04366)
- WMT16 En-Ro (BLEU) scores from He et al. (2021). Full FT 37.3; MAM adapter (6.7%) 37.5; parallel adapter at FFN (12.3%) 37.3; sequential adapter (7.2%) 36.9; LoRA-FFN (7.2%) 36.8; prefix tuning (3.6%) 35.6. — [He et al. 2021 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2110.04366)
- On MNLI/SST2, the MAM adapter at 0.5% of parameters matches full FT (87.4±.3 / 94.2±.3). — [He et al. 2021 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2110.04366)
- **Compacter (2021, NeurIPS):** adapter weights are sums of Kronecker products of shared "slow" matrices and low-rank per-layer "fast" matrices (a parameterized hypercomplex multiplication). "By only training 0.047% of a pretrained model's parameters, Compacter performs on par with standard fine-tuning on GLUE and outperforms standard fine-tuning on SuperGLUE and low-resource settings." — [Mahabadi et al. 2021](https://arxiv.org/abs/2106.04647)
- **AdapterDrop (Rücklé et al., 2020), speed of plain adapters:** adapter training is "up to 60% faster" than full fine-tuning, because no gradients are computed for the frozen weights. Inference is "4–6% slower" because of the extra modules. — [Rücklé et al. 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2010.11918)
- **AdapterDrop, dropping lower-layer adapters:** removing adapters from the first 5 layers keeps 97.1% of performance (specialized AdapterDrop) or 95.4% (robust AdapterDrop), averaged over GLUE. — [Rücklé et al. 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2010.11918)
- **AdapterDrop, speedups:** with 5 dropped layers, inference is 21–42% faster depending on the number of simultaneous tasks (39% faster for 8 tasks), and training steps are 26% faster. Each shared lower layer yields a 4.3–8.4% inference speedup for 2–16 simultaneous tasks. — [Rücklé et al. 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2010.11918)
- **AdapterDrop, pruning AdapterFusion:** with 8 adapters, AdapterFusion trains 47% slower and infers 62% slower than full FT. Pruning to 2 of 8 adapters gives "comparable results" and 68% faster inference. — [Rücklé et al. 2020 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2010.11918)
- In the LLM era (LLaMA-7B, 2023), the best placements were: series adapter after the MLP (59.5% math average), parallel adapter at the MLP (61.7%), and LoRA in both attention and MLP (60%). — [Hu et al. 2023, LLM-Adapters](https://arxiv.org/html/2304.01933)
- Lialin et al. (2023) list trainable-parameter ranges for additive methods: standard adapters 0.1–6%, prompt tuning about 0.1%, prefix tuning 0.1–4%, (IA)^3 about 0.02%. — [Lialin et al. 2023](https://ar5iv.labs.arxiv.org/html/2303.15647)
- Infrastructure: AdapterHub is the project that packages and shares adapters for Transformer models. — [AdapterHub paper, Pfeiffer et al. 2020](https://arxiv.org/abs/2007.07779)

### Inferences
- The evidence converges on one pattern. An adapter placed *in parallel with the FFN* in *every* layer, with a scaling factor, is the strongest classic adapter design. Lower-layer adapters contribute least, and can be dropped for speed at a cost of about 3–5% of relative quality.
- Parameter budgets for matching full FT depend on the task. About 0.05–0.5% is enough for GLUE-style classification, but about 5–7% or more is needed for summarization and translation (He et al. 2021).

### Gaps
- Exact Compacter / Compacter++ per-task GLUE scores and the Compacter++ parameter percentage were not retrieved (only the abstract was fetched).
- MAD-X (language adapters plus invertible adapters, Pfeiffer et al. 2020) was not fetched in this session.

## Appending or stacking whole new transformer blocks on a frozen model (LLaMA Pro, G_stack, LiGO, SOLAR DUS, identity-initialized inserted blocks)

### Takeaway
Adding identity-initialized transformer blocks and training *only* those blocks works for domain adaptation with little forgetting. LLaMA Pro (2024) added 8 blocks to LLaMA2-7B and roughly doubled HumanEval and MBPP scores while general benchmarks moved by at most about ±1 point. Interleaving the new blocks is slightly better than stacking them on top, and stacking at the bottom is catastrophic. The 2025–2026 follow-ups (ADEPT, LESA) choose *which* layers to expand and how to initialize them. By contrast, model-growth methods (G_stack, LiGO, SOLAR) are pretraining accelerators that train *all* weights after growth, so they are not "frozen-backbone" methods.

### Cited Findings
- **LLaMA Pro (Wu et al., 2024; ACL 2024), architecture:** LLaMA2-7B grows from 32 to 40 blocks, giving LLaMA Pro-8.3B. The 32 original blocks are split into 8 groups of 4, and a copy of the top block of each group is inserted after it (interleaved). — [Wu et al. 2024 (HTML)](https://arxiv.org/html/2401.02415)
- **LLaMA Pro, identity initialization:** the attention output projection W^O and FFN output projection W_3 of each new block are zero-initialized. Each new block therefore starts as an identity map through the residual connection, and the expanded model initially reproduces the base model exactly. — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **LLaMA Pro, training:** "Only newly added 8 blocks" are trained; the original 32 blocks are frozen. The corpus is about 80B tokens (Proof-Pile-2 55B math, The-Stack-Dedup Python 22B code), and compute was 2830 H800 GPU-hours (16 GPUs for about 7 days). The abstract states: "We tune the expanded blocks using only new corpus, efficiently and effectively improving the model's knowledge without catastrophic forgetting." — [Wu et al. 2024](https://arxiv.org/html/2401.02415); [abstract](https://arxiv.org/abs/2401.02415)
- **LLaMA Pro-8.3B vs LLaMA2-7B results:**
  - General tasks: ARC 54.10 vs 53.07; HellaSwag 77.94 vs 78.59; MMLU 47.88 vs 46.87; TruthfulQA 39.04 vs 38.76; Winogrande 73.95 vs 74.03.
  - Math and code: GSM8K 17.89 vs 14.48; HumanEval 28.66 vs 13.05; MBPP 33.20 vs 20.09.

  — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **LLaMA Pro placement ablation** (8 blocks, general-task average): interleaved 56.70, stacked on *top* 56.19, stacked at *bottom* 29.36, which is severe degradation. — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **LLaMA Pro method ablation** (general-task average): MoE-style expansion 56.56; full fine-tuning of all blocks 55.01, with "more forgetting." LoRA at rank 1024 scored 58.15 on general ability but was reported to struggle to fit the new-domain distribution. The fetched summary did not make clear whether 58.15 averages general tasks only, so treat it with caution. — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **LLaMA Pro block-count ablation:** adding 1, 2, 4, 8, 16 or 32 blocks, the authors found 8 gave "optimal performance with minimal cost." — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **LLaMA Pro forgetting evidence:**
  - Token distribution on general queries: 82.3% of tokens unshifted and 5.2% significantly shifted.
  - Perplexity: LAMBADA (general) 3.39 → 3.46, Stack (code) 9.46 → 5.25.
  - After identical instruction tuning, LLaMA Pro outperformed LLaMA2 on all tasks.

  — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **ADEPT (2025; ICLR 2026)** is a two-stage domain-adaptive continual pretraining framework.
  - Stage 1, "General-Competence Guided Selective Layer Expansion," duplicates the layers *least critical for the general domain*.
  - Stage 2, "Adaptive Unit-Wise Decoupled Tuning," splits parameter units inside the expanded layers by general-domain importance and gives them asymmetric learning rates.
  - Versus full-parameter continual pretraining it reports up to +5.58% accuracy on target-domain benchmarks and up to +5.76% on the general domain, with only 15% of parameters tuned and less training time.

  — [ADEPT, arXiv 2510.10071](https://arxiv.org/abs/2510.10071); [ICLR 2026 listing](https://mlanthology.org/iclr/2026/zhang2026iclr-adept/)
- **ADEPT's motivation:** uniform layer expansion as in LLaMA Pro "still entangle[s] general and domain learning." — [ADEPT](https://arxiv.org/html/2510.10071v1)
- **LESA (2025):** instead of copying layers heuristically, it concatenates existing layers' parameters, applies SVD, and trains a network to *predict* the parameters of layers inserted between adjacent layers. It reports "superior performance with less than half the computational cost during continual pre-training" compared with heuristic depth-scaling baselines. — [LESA, arXiv 2502.13794](https://arxiv.org/abs/2502.13794)
- **Frozen-substrate growth ("Growing Transformers," 2025, revised May 2026):** a preprint that stacks new blocks while keeping earlier blocks "sealed," training only the newest layers and the LM head on top of frozen token embeddings. Active trainable parameters stay roughly constant: 105.0M active versus 180.5M for a monolithic baseline in a 9-layer study. A 16-layer 269.7M model trained on 68.9B tokens reached only 28.92% MMLU after a LoRA integration phase. The authors acknowledge "a clear tradeoff against dense monolithic training in final perplexity." — [arXiv 2507.07129](https://arxiv.org/abs/2507.07129)
- **G_stack (2024; NeurIPS 2024 Spotlight):** a *pretraining* acceleration method (depthwise stacking of a small trained model), not a frozen-backbone method. It reached the baseline's loss with 194B tokens instead of 300B, a 54.6% speedup. It scaled to 7B models and 750B tokens and gave improvements on 8 NLP benchmarks, with guidelines for growth timing and growth factor. — [Du et al. 2024, "Stacking Your Transformers"](https://arxiv.org/abs/2405.15319)
- **LiGO (2023; ICLR 2023):** learns a linear growth operator, factorized into width and depth operators with Kronecker structure, that maps a small model's parameters to initialize a larger one. It saves up to 50% of the compute of training from scratch on BERT, RoBERTa, GPT-2 and ViT, and beats other small-to-large reuse baselines. — [Wang et al. 2023](https://arxiv.org/abs/2303.00980)
- **SOLAR 10.7B depth up-scaling (DUS, Dec 2023):**
  - Construction: the 32-layer Llama-2 architecture is initialized with Mistral 7B weights. The last 8 layers of one copy and the first 8 layers of a duplicate are removed, and the two 24-layer halves are concatenated into 48 layers (10.7B params).
  - Training: continued pretraining is *full-parameter*. Performance drops at first and then recovers rapidly, which the authors attribute to the reduced "layer distance" at the seam.
  - Results: H6 average 66.04 for SOLAR 10.7B versus 60.97 for Mistral 7B. The Instruct model scores 74.20 versus 72.62 for Mixtral-8x7B-Instruct.

  — [Kim et al. 2023 (HTML)](https://arxiv.org/html/2312.15166)
- **Why duplicating or inserting middle layers is benign ("Transformer Layers as Painters," 2024):** "the lower and final layers of pretrained transformers differ from middle layers, but … middle layers have a surprising amount of uniformity." Models tolerate skipping, reordering and parallel execution of middle layers, while the first and last layers are specialized. — [Sun et al. 2024](https://arxiv.org/abs/2407.09298)

### Inferences
- The LLaMA Pro placement ablation is the cleanest single data point for the user's question. Blocks stacked on top of a frozen LLM work nearly as well as interleaved ones for preserving general ability (56.19 vs 56.70). Blocks at the bottom (29.36) break the model, likely because every downstream frozen layer then sees shifted inputs.
- This is consistent with the "Painters" finding that middle layers share a representation space and tolerate duplication.
- Identity initialization is what makes block insertion safe on a frozen model: with zero-init output projections, training starts exactly at the base model's function. Without it, the frozen layers after an inserted block would receive out-of-distribution inputs.
- The cost is permanent inference overhead. LLaMA Pro has 25% more layers (40 vs 32) and about 19% more parameters (8.3B vs 7B). The latency figure is inferred from layer count and was not measured in the source.
- The model-growth line (G_stack, LiGO, SOLAR DUS) shows that duplicated layers are a good *initialization*. However, those methods then unfreeze everything, so they are evidence about initialization rather than about frozen-backbone adaptation.

### Gaps
- No verified per-benchmark code/math numbers were retrieved for LLaMA Pro's top-stacking ablation. Only the general-task average (56.19) was obtained, so it is unknown whether top-only stacking learns the new domain as well as interleaving.
- ADEPT's exact per-benchmark numbers and base models, and whether its original (non-expanded) layers are strictly frozen, were not retrieved: the ICLR PDF was too large to fetch.
- LESA's frozen-versus-unfrozen training protocol was not stated in the abstract.

## Side networks that run beside a frozen backbone (Side-Tuning, LST, Res-Tuning, HST, QST, Ladder/xLadder, proxy-tuning)

### Takeaway
Side networks keep the backbone in pure inference mode, with no backpropagation through it. This cuts training memory by about 50–70% versus full FT or QLoRA, at a cost of roughly 0–1 point of accuracy (LST: 84.1 vs 85.2 GLUE; QST: 63.9 vs 64.1 MMLU on 70B). The idea started with Side-Tuning (2019), was made competitive for NLP by LST (2022), was scaled to 70B LLMs by QST (2024), and was re-validated against QLoRA on 7B models in late 2025. Proxy-tuning (2024) is the extreme case: the "added module" is an entire small model whose logit difference steers a frozen large model at decode time.

### Cited Findings
- **Side-Tuning (Zhang et al., 2019; vision-first, but the canonical source):** a frozen base network B(x) is combined with a trainable side network S(x) as R(x) = α·B(x) + (1−α)·S(x), with α learnable. — [Zhang et al. 2019 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1912.13503)
  - The base model "remains completely unchanged," which prevents catastrophic forgetting.
  - Side-network size scales with task difficulty rather than base-model size.
  - It was evaluated on iCIFAR, Taskonomy, **SQuAD v2 with a pretrained BERT**, and Habitat navigation, with average rank 1.33 of 6 on iTaskonomy incremental learning.
- **Ladder Side-Tuning (LST; Sung, Cho, Bansal, 2022), design:** a side network 1/r the width of the backbone (r = 8 or 16) receives *downsampled intermediate activations from every backbone layer* ("ladder" shortcuts). A learned gate μ_i = sigmoid(α_i/T) mixes each backbone activation with the previous side layer. The side network is initialized by structurally pruning backbone weights using Fisher information, and alternate side layers can be dropped. — [Sung et al. 2022 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2206.06522)
- **LST on T5-base GLUE:**
  - Accuracy: full FT 85.2, Adapters 85.3, LoRA 85.3, LST 84.1.
  - Training memory: full FT 17.6 GB, Adapters 13.0 GB, LoRA 12.6 GB, LST 5.5 GB, which is a 69% reduction versus full FT and 2.7× more saving than adapters or LoRA.
  - Parameters: LST 1.74%, Adapters 1.63%, LoRA 1.71%.

  — [Sung et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.06522)
- **LST at scale and on vision-language:** T5-large 87.1 GLUE at 12.2 GB; T5-3B 88.1 at 22.4 GB, which outperforms the other PETL methods. On CLIP-T5 vision-language tasks, 76.9 average at 15.3 GB versus 28.4 GB for adapters. Inference memory is 0.88 GB versus 0.86 GB for full FT. — [Sung et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.06522)
- **Res-Tuning (2023; vision and diffusion):** "unbinds" tuners from the backbone so they run in parallel with frozen operations. In the Res-Tuning-Bypass variant, "gradients are back-propagated only to the tuners but not to the backbone." — [Jiang et al. 2023 (HTML)](https://arxiv.org/html/2310.19859)
  - Memory: −49.7% for discriminative tasks and −70.7% for Stable Diffusion v1.5, with 58.6% less training time for text-to-image and an 80.9–83.6% multi-task inference speedup.
  - Accuracy: CIFAR-100 93.25% with 0.55% of parameters versus 89.12% for full FT.
- **Hierarchical Side-Tuning (HST, 2023; ViT):** a lightweight hierarchical side network over intermediate activations with 0.78M trainable parameters. It achieves 76.1% average top-1 on VTAB-1K, state of the art on 13 of 19 tasks, and surpasses full FT on COCO and ADE20K dense prediction. — [Lin et al. 2023](https://arxiv.org/abs/2310.05393)
- **Quantized Side Tuning (QST, 2024):** — [Zhang et al. 2024 (HTML)](https://arxiv.org/html/2401.07159)
  - Design: the LLM is quantized to 4-bit NF4 and a side network with r-times smaller hidden size consumes downsampled LLM hidden states, avoiding backpropagation through the LLM. The downsamplers are adapters, LoRA, or max/avg pooling.
  - Efficiency: up to 2.3× memory reduction versus QLoRA and up to 7× versus full FT, with about 3× faster training.
  - Accuracy: LLaMA-2-70B MMLU 5-shot 63.9 at 56.0 GB versus 64.1 at 95.5 GB for QLoRA; GLUE generally within 1–2 points of QLoRA. Evaluated on OPT 1.3B–66B and LLaMA-2 7B–70B.
- **"Ladder Up, Memory Down" (Dec 2025, revised Aug 2026):** LST on modern 7B LLMs "cuts peak memory by 50%" versus QLoRA. — [arXiv 2512.14237](https://arxiv.org/abs/2512.14237)
  - It enables "fine-tuning of 7B-parameter models on a single 12 GB consumer GPU with 2k-token contexts, requiring no gradient checkpointing," a setting where QLoRA runs out of memory.
  - Accuracy is "competitive with QLoRA's accuracy on average" on NLU, math and LLM-critic tasks, and LST "matches QLoRA's compute scaling slope."
  - It introduces **xLadder**, a depth-extended variant that increases effective depth via cross-connections and "shortens chain-of-thought (CoT) at fixed parameter count."
- **MobiLLM (2025):** applies side tuning to on-device LLM fine-tuning. It "separates a set of parallel adapters from the backbone to create a backpropagation bypass," with the server handling the side network and only one-way activation transfers. — [MobiLLM, arXiv 2502.20421](https://awesomepapers.io/systems-efficiency/papers/2502.20421); [ACM EMFM workshop](https://dl.acm.org/doi/10.1145/3737902.3768351)
- **Proxy-tuning (Liu et al., 2024):** at decode time the logits are base_large + (tuned_small − untuned_small), with no access to the large model's weights. — [Liu et al. 2024 (HTML)](https://arxiv.org/html/2401.08565)
  - Headline result: with 7B experts on Llama2-70B, it "closes 88% of the performance gap" to Llama2-70B-chat.
  - On open-ended TruthfulQA it *beats* direct tuning (92.3 vs 85.8 at 70B), and it improves GSM by 44.3 points at 70B.
  - Runtime is about 2.4× (13B) and 1.5× (70B) because the models run sequentially.
  - Its influence falls mainly on reasoning and style tokens rather than factual content.

### Inferences
- The defining advantage of side networks is training memory: no activations or gradients are stored for the backbone. Their accuracy ceiling is set by how rich the taps into the backbone are. LST, which reads *every* layer's activations, lands within about 1 point of full FT, while the original Side-Tuning, which sums only at the output, was mainly shown on vision and incremental learning.
- Side networks are an intermediate case between "top-only appending" and "inserting throughout depth." They read from all depths but cannot *change* the frozen backbone's internal computation, which may explain their small but consistent gap versus adapters and LoRA on GLUE (84.1 vs 85.3).
- Proxy-tuning shows that even with *zero* access to the large model's internals, an external module can move behavior most of the way (88%). It helps most on style and format and least on injecting new knowledge.

### Gaps
- "SAN" and "LoSA" named in the assignment could not be verified in this session. I believe SAN is the Side Adapter Network for open-vocabulary segmentation (CVPR 2023) and LoSA a long-short-range side adapter for temporal action localization; both are vision/video. They are listed here as unverified pointers only.
- The xLadder benchmark numbers (accuracy, CoT length reduction) in arXiv 2512.14237 were not retrieved beyond the abstract.
- No 2026 frontier-scale (>100B) side-tuning results were found.

## Appending only at the top vs. inserting throughout depth (and evidence from probing intermediate layers)

### Takeaway
Top-only appending on frozen final hidden states is consistently weaker than depth-distributed insertion. Four lines of evidence support this. Tuning only the output layer recovers about 64% of full-FT quality on GLUE (2019). Matching adapters by tuning only the top layers needs about 50× more parameters per task (Houlsby 2019). Frozen features plus a head fail on fine-grained or cross-sentence tasks (Liu et al. 2019; Peters et al. 2019). Final layers are often *not* the best representation: intermediate layers win on 32 embedding tasks (ICML 2025). The exception is appending *full transformer blocks* on top of a frozen LLM with identity init, which preserves general ability nearly as well as interleaving (LLaMA Pro).

### Cited Findings
- **Freezing all but the output layer (BERT/RoBERTa, GLUE):** gives about 64% of full-model quality on average. "Only a fourth of the final layers need to be fine-tuned to achieve 90% of the original quality": 3–5 of 12 layers for base models, and about 6 (BERT) or 7 (RoBERTa) of 24 for large models. The first half of the model can be frozen. — [Lee et al. 2019, "What Would Elsa Do?"](https://ar5iv.labs.arxiv.org/html/1911.03090)
- Houlsby et al. (2019) compared adapters with "variable fine-tuning" of the top n layers. Matching quality needed 52.9% of parameters per task for top-layer tuning versus 1.14% for adapters inserted throughout depth. — [Houlsby et al. 2019](https://ar5iv.labs.arxiv.org/html/1902.00751)
- **Feature extraction vs fine-tuning ("To Tune or Not to Tune?", 2019):** this compares a frozen encoder plus a task-specific architecture on top against fine-tuning. — [Peters, Ruder, Smith 2019 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1903.05987)
  - For BERT, fine-tuning wins by +2.3 to +6.7 points on sentence-pair tasks and is roughly equal (+0.5) on NER and sentiment.
  - For ELMo, frozen features are equal or better.
  - The analysis says single-sentence tasks concentrate task information in the final layers, while sentence-pair tasks build it gradually across intermediate layers.
- **Linear probes on frozen contextual representations (17 tasks, 2019):** probes are competitive with task-specific SOTA on many tasks but fail on tasks needing "fine-grained linguistic knowledge (e.g., conjunct identification)." — [Liu et al. 2019](https://ar5iv.labs.arxiv.org/html/1903.08855)
  - Adding task-specific contextualization (an LSTM on top) substantially helps the hard tasks.
  - For transformers, no single layer is best, middle layers transfer best, and scalar mixing of layers beats any single layer.
- **"Layer by Layer" (Skean et al., 2025; ICML 2025):** mid-depth embeddings consistently outperform final-layer embeddings across 32 text-embedding tasks, for transformers and state-space models. The explanation offered is that intermediate layers better balance compression against signal preservation (information-theoretic, geometric and invariance metrics). — [Skean et al. 2025](https://arxiv.org/abs/2502.02013)
- **Adapter ablations show adaptation concentrates in upper-middle layers, not only the top.** Removing adapters from layers 0–4 barely hurts MNLI (Houlsby 2019), and dropping adapters from the first 5 layers retains 95–97% of performance (AdapterDrop). — [Houlsby et al. 2019](https://ar5iv.labs.arxiv.org/html/1902.00751); [Rücklé et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.11918)
- **LLaMA Pro placement ablation** (8 identity-initialized blocks, general-task average): interleaved 56.70, top 56.19, bottom 29.36. — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **Robustness side of the trade-off (vision, but the canonical theory paper):** full fine-tuning is about 2% better in-distribution than linear probing but about 7% worse out-of-distribution across 10 distribution-shift datasets. The mechanism is "feature distortion" when lower layers update. LP-FT (probe first, then fine-tune) is 1% better ID and 10% better OOD than full FT. — [Kumar et al. 2022](https://arxiv.org/abs/2202.10054)

### Inferences
- *Why depth-wise insertion beats a top-only head:*
  - A head on frozen final states can only re-read what the final layer kept, and final layers of pretrained LMs are specialized for next-token or pretraining objectives (Painters, Layer by Layer).
  - Some task-relevant structure lives in, or is best expressed at, intermediate layers (Liu 2019; Skean 2025).
  - Tasks needing new cross-token interactions (sentence pairs) must alter how intermediate layers attend (Peters 2019).
  - Depth-wise modules can also re-route computation at every layer, so downstream frozen layers process *adapted* inputs. Top-only modules cannot do this.
- A top-appended *transformer block* is much stronger than a pooled classification head, because it still attends over all token positions. That may explain why LLaMA Pro's top stacking is close to interleaving on general ability. It does not show that top stacking learns the new domain as well as interleaving; that evidence was not retrieved.
- Practical design implication: if appending must stay at the top, feed it multiple intermediate layers (a scalar mix or ladder taps as in LST) rather than only the last hidden state.

### Gaps
- No direct LLM-era (≥7B decoder) controlled study was found comparing "k new transformer layers on top of frozen final states" against "the same parameter budget in adapters distributed through depth" on generative tasks. Current evidence is from BERT-era NLU or from LLaMA Pro's general-ability-only ablation.
- Classification heads on frozen embeddings are covered by another researcher, so they were not explored here.

## Theoretical and expressivity arguments: what can an appended module on frozen features represent vs. what requires modifying internal computation?

### Takeaway
Theory says input-side additive methods (prompts and prefixes) cannot change the *relative* attention pattern over content. They can only bias attention outputs in a fixed direction, so they elicit existing skills rather than learning new attention patterns (Petrov et al. 2023). Universality *is* achievable in principle, but only with large prefixes and depth that grows with sequence length (Petrov et al. 2024). A module on top of frozen final features is limited to functions of those features, so information discarded by the backbone cannot be recovered. Modules inserted inside layers (adapters, new blocks) avoid that bottleneck because they alter the inputs of every downstream frozen layer.

### Cited Findings
- **"When Do Prompting and Prefix-Tuning Work?" (Petrov, Torr, Bibi, 2023):** context-based fine-tuning "cannot change the relative attention pattern over the content." It can "only bias the outputs of an attention layer in a fixed direction," and is "potentially less expressive than full fine-tuning, even with the same number of learnable parameters." These methods elicit skills present in the pretrained model but "may not be able to learn novel tasks that require new attention patterns." — [Petrov et al. 2023](https://arxiv.org/abs/2310.19698)
- **Universal approximation via prompting (Petrov, Torr, Bibi, 2024):** "prompting and prefix-tuning a pretrained model can universally approximate sequence-to-sequence functions," and even "a single attention head" suffices for continuous functions. However, general sequence-to-sequence functions need transformer depth that scales linearly with sequence length, and Jackson-type bounds quantify the prefix length needed for a given accuracy. — [Petrov et al. 2024](https://arxiv.org/abs/2402.14753)
- **Unified-view theory (He et al. 2021):** all of adapters, prefix tuning and LoRA are hidden-state modifications Δh added at some position. They differ in whether Δh is computed from the sublayer input or output (parallel vs sequential), at which sublayer (attention vs FFN), and with what scaling. Empirically, larger-capacity FFN-side parallel modification is most expressive per parameter. — [He et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.04366)
- **Empirical expressivity limit of frozen features plus a head:** linear probes fail on fine-grained linguistic tasks, and adding contextualization layers on top helps. — [Liu et al. 2019](https://ar5iv.labs.arxiv.org/html/1903.08855)
- **Feature-distortion theory** is the counter-argument: *not* modifying lower layers can be better out of distribution, so frozen-feature approaches trade some ID accuracy for robustness. — [Kumar et al. 2022](https://arxiv.org/abs/2202.10054)
- **Proxy-tuning evidence:** an external module steering only output logits mainly influences reasoning and stylistic tokens rather than factual content. — [Liu et al. 2024](https://arxiv.org/html/2401.08565)

### Inferences
- The following is a formal restatement of what the cited results imply; no single source proves it for LMs.
  - A top-only module g computes g(h_L(x)). Any distinction the backbone collapses at layer L is unrecoverable (data-processing inequality).
  - An inserted module at layer ℓ changes h_ℓ, so all frozen layers above it compute different functions. The frozen backbone thus acts as a fixed "library" of transformations that inserted modules can re-route.
  - Side networks with ladder taps can read h_1…h_L, removing the information bottleneck, but they still cannot make frozen layers attend differently.
- The ordering of expressivity implied by the evidence, from least to most, is:
  1. top-only head on final features;
  2. prompt or prefix (fixed-direction attention bias);
  3. side network reading all layers;
  4. adapters or new blocks inserted inside the forward path;
  5. weight modification (LoRA or full FT).

  This roughly matches the empirical accuracy ordering on generation tasks in He et al. 2021, where prefix < sequential adapter < parallel/MAM ≈ full FT.
- An inserted identity-initialized transformer block is expressive in a way adapters are not: it has its own attention and can therefore create *new* attention patterns. This may be why block expansion suits injecting new domains (code, math) better than prompt-style methods do.

### Gaps
- No formal expressivity theorem specific to *bottleneck adapters* or *inserted transformer blocks* on a frozen backbone was retrieved; LoRA expressivity theory is covered by another researcher.
- No theoretical treatment was found of when top-stacked blocks versus interleaved blocks differ in representational power.

## Known failure modes and limitations of additive approaches

### Takeaway
The main costs are:
- inference latency from added sequential depth (+20–30% at batch size 1 for sequential adapters; permanent +25% layers for LLaMA Pro);
- a residual quality gap on hard generation tasks at small parameter budgets;
- a small accuracy gap for memory-optimized side networks;
- 1.5–2.4× decode slowdown for logit-arithmetic methods;
- sensitivity to *where* modules go (bottom-stacked blocks are catastrophic);
- slowdowns in fusion-style multi-adapter composition (AdapterFusion inference 62% slower).

### Cited Findings
- **Sequential adapter latency (LoRA paper, 2021):** GPT-2 medium single forward pass. — [Hu et al. 2021, LoRA (Table 1)](https://ar5iv.labs.arxiv.org/html/2106.09685)
  - Batch 1: FT/LoRA 19.8 ms, Adapter^L 23.9 ms (+20.7%), Adapter^H 25.8 ms (+30.3%).
  - Batch 16: +5.0% and +8.4%.
  - Batch 32: +2.2% and +3.0%.
  - Adapter layers "have to be processed sequentially," and with model sharding the "additional depth requires more synchronous GPU operations such as AllReduce and Broadcast."
- **Prefix/prompt methods consume context:** "reserving a part of the sequence length for adaptation necessarily reduces the sequence length available to process a downstream task." — [Hu et al. 2021](https://ar5iv.labs.arxiv.org/html/2106.09685)
- **Adapter overhead in the survey literature:** additive methods introduce inference cost, since adapters add FFN computations and prompt or prefix tuning lengthens inputs. — [Lialin et al. 2023](https://ar5iv.labs.arxiv.org/html/2303.15647)
- **Adapter inference is 4–6% slower than a fully fine-tuned model** (BERT-era measurements). AdapterFusion with 8 adapters is 47% slower in training and 62% slower in inference than full FT. — [Rücklé et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.11918)
- **Generation-task gap at small budgets.** On XSum and MT a "large gap is still present" for existing PEFT methods even with about 5–10% or more parameters, while GLUE-style tasks close with 0.5%. XSum ROUGE-2: prefix tuning 20.46 and sequential adapter 20.89 versus full FT 21.94. Only the MAM adapter at 6.7% closes the gap (21.90). — [He et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.04366)
- **Side-network accuracy gap:** LST scores 84.1 versus 85.2 for full FT and 85.3 for adapters and LoRA on T5-base GLUE. QST is within 1–2 points of QLoRA on GLUE and 0.2 points below it on 70B MMLU. — [Sung et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.06522); [Zhang et al. 2024](https://arxiv.org/html/2401.07159)
- **Wrong placement of new blocks breaks the model:** bottom-stacked blocks in LLaMA Pro give a general-task average of 29.36 versus 56.70 for interleaved. — [Wu et al. 2024](https://arxiv.org/html/2401.02415)
- **Uniform block expansion still entangles general and domain learning.** This is the motivation for ADEPT's selective expansion. — [ADEPT 2025](https://arxiv.org/html/2510.10071v1)
- **Frozen-substrate growth trails dense training in perplexity:** "a clear tradeoff against dense monolithic training in final perplexity," and a 269.7M 16-layer model reached only 28.92% MMLU. — [arXiv 2507.07129](https://arxiv.org/abs/2507.07129)
- **Logit-arithmetic side models slow decoding:** proxy-tuning runtime is about 2.4× at 13B and 1.5× at 70B when run sequentially. It closes 88% of the tuning gap, not 100%. — [Liu et al. 2024](https://arxiv.org/html/2401.08565)
- **Model-growth (DUS) needs full-parameter continued pretraining to recover from the initial drop after layer duplication.** It is not a frozen-backbone method. — [Kim et al. 2023, SOLAR](https://arxiv.org/html/2312.15166)
- **Mitigations that exist:**
  - AdapterDrop, dropping lower-layer adapters: 21–42% faster multi-task inference while keeping 95–97% of performance. — [Rücklé et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.11918)
  - Parallel instead of sequential placement, which is also more accurate. — [He et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.04366)
  - LST inference memory is nearly identical to full FT (0.88 vs 0.86 GB). — [Sung et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.06522)

### Inferences
- The latency penalty is structural. Any added module that cannot be algebraically merged into existing weights increases per-token compute and synchronization; the batch-1 online-serving case is worst. This is the main practical reason mergeable reparameterization (LoRA) became the default for single-task LLM deployment (LoRA detail is covered by another researcher).
- Additive methods remain attractive in four situations:
  - many tasks must share one frozen model with exact base-model recovery (adapter libraries, AdapterFusion);
  - training memory is the binding constraint (LST, QST, Ladder 2025);
  - new domain knowledge must be added without forgetting (LLaMA Pro, ADEPT);
  - the large model's weights are inaccessible (proxy-tuning).

### Gaps
- No 2025–2026 measurement was found of LLaMA Pro–style block expansion *latency* at serving time (tokens/s versus base) or of how it interacts with KV-cache size. Per-layer KV cache grows with added layers; this is an inference, not measured.
- No systematic 2025–2026 study was found of additive methods on long-form reasoning or agentic tasks versus full FT. The only related item is the xLadder abstract claim of shorter CoT.
