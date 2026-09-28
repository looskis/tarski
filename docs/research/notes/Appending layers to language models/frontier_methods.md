# Frontier Methods for Adapting LLMs Without Touching Base Weights (2023–2026)

Scope note: this file covers "don't touch the base weights" adaptation beyond vanilla LoRA and classic adapters: representation interventions, SAE/probe classifiers, hypernetwork-generated adapters, memory/KV-cache modules, composition/merging, prompt-based parameters, tiny-parameter PEFT, test-time training, and adapter portability across base-model upgrades. Classic additive architectures (Houlsby adapters, LLaMA Pro block expansion, ladder side-tuning), frozen-embedding classifier heads (linear probes, SetFit, LLM2Vec), and storage/compute economics are covered by other researchers and appear here only for context. Research date: 2026-09-27. Items marked "title only" were seen in search results but not read in full, so only their existence and stated topic are reliable.

## 1. Representation/activation interventions (ReFT/LoReFT, steering vectors, RepE, ActAdd, in-context/function/task vectors): how do they compare to LoRA, including on classification?

### Takeaway
Trained representation interventions (LoReFT, April 2024) roughly match LoRA on classification (GLUE) and commonsense tasks with 15–65x fewer parameters. They are weaker on long chain-of-thought generation. Untrained or lightly trained steering vectors are cheap but unreliable. Prompting still beats them for steering (AxBench, Jan 2025). For concept detection, simple supervised directions (DiffMean, linear probe, ReFT-r1) reach about 0.94 AUROC. In-context/task vectors (I2CL, ELICIT, 2024–2025) give few-shot-level accuracy at zero-shot cost. That makes them a natural fit for classification and for per-task "task IDs" on one shared frozen backbone.

### Cited Findings
**ReFT / LoReFT (Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts; arXiv v1 2024-04-04, v3 2024-05-22)**
- Mechanism: the base model stays frozen. The method learns task-specific interventions on hidden representations. LoReFT edits a low-rank linear subspace of the hidden states at chosen layers and positions instead of changing weights. The authors call it "15x–65x more parameter-efficient than LoRA" and a drop-in replacement for PEFTs — [ReFT arXiv abs](https://arxiv.org/abs/2404.03592)
- GLUE with RoBERTa-base: full FT 85.6 avg; LoRA (0.239% params) 84.7; LoReFT (0.015%) 84.2; DiReFT (0.015%) 83.2. With RoBERTa-large: full FT 88.6; LoRA (0.225%) 88.1; LoReFT (0.014%) 88.2; DiReFT 87.4 — [ReFT paper HTML v3](https://arxiv.org/html/2404.03592v3)
- Commonsense reasoning with LLaMA-7B: LoRA (0.826%) 74.7 vs LoReFT (0.031%) 80.2. With LLaMA-13B: LoRA (0.670%) 80.5 vs LoReFT (0.025%) 83.3 — [ReFT paper HTML v3](https://arxiv.org/html/2404.03592v3)
- Weakness: on arithmetic reasoning (math word problems), "both LoReFT and DiReFT do not perform as well" as LoRA. The authors attribute this to long chain-of-thought generation. They list limitations: few non-LLaMA families were tested, the hyperparameter search space is large, and it is unclear why interventions work at scale — [ReFT paper HTML v3](https://arxiv.org/html/2404.03592v3)

**AxBench (Wu et al.; arXiv 2025-01; ICML 2025 poster; one secondary note labels it an ICLR'25 spotlight, so the venue is inconsistent across sources)**
- Steering: prompting was the clear winner with an average overall score of 0.894. Fine-tuning-style methods followed (LoReFT 0.741, SFT 0.676, LoRA 0.615) and generally beat representation-steering methods. Among the representation methods, ReFT-r1 (rank-1) was the most competitive — [AxBench alphaXiv summary](https://www.alphaxiv.org/abs/2501.17148); [arXiv abs](https://arxiv.org/abs/2501.17148); [ICML 2025 poster](https://icml.cc/virtual/2025/poster/45658)
- Concept detection, which works like a classifier: mean AUROC was DiffMean 0.942, linear probe 0.940, ReFT-r1 0.938. Vanilla SAE features scored 0.695, and SAE-A (AUROC-selected features) 0.917 — [AxBench alphaXiv summary](https://www.alphaxiv.org/abs/2501.17148)

**HyperSteer (Sun, Baskaran, Wu, Sklar, Potts, Geiger; arXiv 2025-06-03)**
- A hypernetwork trained end-to-end generates steering vectors from natural-language steering prompts plus the steered LM's internals. It scales to "thousands of steering prompts". It beats prior activation-steering methods even on unseen prompts, and it performs "on par with steering-via-prompting" — [HyperSteer arXiv](https://arxiv.org/abs/2506.03292)

**Steering reliability (2024–2026)**
- Steering vectors often generalise, but for several concepts they are brittle to reasonable prompt changes and fail out of distribution. Spurious biases can drive much of per-input steering effectiveness — [Tan et al., "Analysing the Generalisation and Reliability of Steering Vectors" (OpenReview)](https://openreview.net/forum?id=v8X70gTodR)
- Steering effects "vary significantly across behaviors, and are often unreliable or even counterproductive" — [Steering off Course (arXiv 2504.04635, Apr 2025)](https://arxiv.org/pdf/2504.04635)
- Position paper (ACL 2026 main; arXiv 2026-04-15): it argues that activation steering should be treated as its own adaptation paradigm ("targeted interventions in activation space") alongside fine-tuning, PEFT and prompting. It proposes a unified functional taxonomy. The abstract gives no quantitative comparison — [From Weights to Activations (arXiv 2604.14090)](https://arxiv.org/abs/2604.14090)
- Title-only 2026 follow-ups show where the field is going: *Steer2Adapt: Dynamically Composing Steering Vectors Elicits Efficient Adaptation of LLMs* (Feb 2026) — [arXiv 2602.07276](https://arxiv.org/pdf/2602.07276); *Inverted Detection and Control in Steering Vectors* (Aug 2026) — [arXiv 2608.02957](https://arxiv.org/pdf/2608.02957); *Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions* (May 2026) — [arXiv 2605.05983](https://arxiv.org/pdf/2605.05983); *The Geometric Canary: Predicting Steerability and Detecting Drift via Representational Stability* (Apr 2026) — [arXiv 2604.17698](https://arxiv.org/pdf/2604.17698); *KV Cache Steering for Controlling Frozen LLMs* (Jul 2025) — [arXiv 2507.08799](https://arxiv.org/pdf/2507.08799)

**Foundational steering and representation-engineering papers (context, 2023)**
- Activation Addition (ActAdd): steering by adding activation differences from contrasting prompt pairs at inference — [Turner et al., arXiv 2308.10248](https://arxiv.org/abs/2308.10248)
- Representation Engineering (RepE): reading and controlling high-level concepts via population-level representation directions — [Zou et al., arXiv 2310.01405](https://arxiv.org/abs/2310.01405)
- Survey of RepE methods and research challenges (Feb 2025) — [Bartoszcze et al., arXiv 2502.17601](https://arxiv.org/html/2502.17601v1)

**In-context vectors, function vectors and task vectors (2023–2025)**
- Function vectors: a small set of attention heads at the last prompt token output vectors that encode the demonstrated task. Summing them triggers the task in zero-shot contexts. Later work finds that function-vector heads matter more to ICL than induction heads — [ELICIT arXiv 2410.09343 (related-work summary)](https://arxiv.org/pdf/2410.09343); original: [Todd et al., arXiv 2310.15213](https://arxiv.org/abs/2310.15213); [Hendel et al., "In-Context Learning Creates Task Vectors", arXiv 2310.15916](https://arxiv.org/abs/2310.15916)
- I2CL (Implicit In-context Learning; ICLR 2025): builds a "context vector" from demonstrations and injects a linear combination of it with the query activations into the residual stream. It reaches few-shot-level accuracy at zero-shot inference cost. It beats zero-shot by an average of 17 points absolute across nine real-world tasks on Llama2-7B, and it is robust to bad demonstrations: when poor demos drop ICL by 7 points, I2CL drops 0.5. The context vectors also serve as "task-ids" for task-similarity detection and transfer — [I2CL arXiv 2405.14660](https://arxiv.org/abs/2405.14660); [OpenReview](https://openreview.net/forum?id=G7u4ue6ncT)
- ELICIT (arXiv Oct 2024): a "capability library" of task vectors, each condensing one ICL-learned capability. A retrieval module decides when to inject a vector — [ELICIT arXiv 2410.09343](https://arxiv.org/pdf/2410.09343)
- Vector-ICL (ICLR 2025): lightweight projectors map continuous vectors (e.g., from other encoders) into the LLM embedding space so the frozen LLM can do ICL over them. Pretraining the projectors with an LM objective enables this, and task fine-tuning of the projector improves it further — [Vector-ICL arXiv 2410.05629](https://arxiv.org/abs/2410.05629)

### Inferences
- For classification specifically, LoReFT's GLUE parity with LoRA at about 1/16 the parameters, together with AxBench's finding that simple supervised directions reach about 0.94 AUROC on concept detection, suggests the following for routing and ticket categorization: a rank-1 to rank-8 intervention or a DiffMean/probe direction per label is probably already close to the accuracy ceiling of weight-based PEFT. The costs and weaknesses show up on generation-heavy tasks (math CoT, open-ended steering), not on classification.
- Representation interventions are position- and layer-local. That means many task interventions can in principle run on one shared frozen forward pass, which fits one backbone serving many classifiers. However, none of the sources read reports a systematic multi-intervention, single-pass serving benchmark.
- I2CL's "task-id" property and ELICIT's retrieval over task vectors are close conceptual cousins of message routing: the routing decision can itself be read from, or made in, activation space.

### Gaps
- No head-to-head benchmark was found that compares LoReFT, DiffMean/steering directions, I2CL task vectors, LoRA and probes on realistic many-class intent/routing datasets (e.g., Banking77, CLINC150) with the same frozen backbone. GLUE is binary or small-label-space.
- Whether ReFT interventions survive base-model upgrades has not been studied in any source found (see Q9).
- The exact per-model breakdown behind AxBench's aggregate steering scores was not verified; the numbers above come from a summary page.

## 2. Probing and SAE features as classifiers: do they beat fine-tuning?

### Takeaway
The weight of evidence (2025) is that SAE-latent probes do not reliably beat simple baselines such as logistic regression or DiffMean on raw activations. Neither SAEs nor probes have been shown to beat fine-tuning in general. One EMNLP 2025 paper reports SAE features beating hidden-state baselines on safety classification, with cross-model transfer, so the literature is contested. The practical value of SAEs for classification is interpretability, auditing and spurious-feature detection, not raw accuracy.

### Cited Findings
- "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda; arXiv 2502.16681, Feb 2025; ICML 2025). SAE probes were tested under data scarcity, class imbalance, label noise and covariate shift. SAEs occasionally win on individual datasets, but the authors could not build SAE+baseline ensembles that consistently beat baseline-only ensembles. The apparent SAE advantages (finding spurious correlations, detecting poor dataset quality, multi-token probes) can be matched by non-SAE baselines — [arXiv PDF](https://arxiv.org/pdf/2502.16681); [PMLR](https://proceedings.mlr.press/v267/kantamneni25a.html)
- AxBench concept detection: vanilla SAE features 0.695 mean AUROC and SAE-A 0.917, vs DiffMean 0.942 and linear probe 0.940 — [AxBench alphaXiv](https://www.alphaxiv.org/abs/2501.17148)
- Contrasting result, EMNLP 2025 ("Sparse Autoencoder Features for Classifications and Transferability"; Gallifant et al.): SAE-derived features reach macro F1 > 0.8 on safety-critical classification. They outperform hidden-state and bag-of-words baselines, transfer from Gemma 2 2B to Gemma 2 9B-IT, and generalize zero-shot to cross-lingual toxicity and visual classification. Binarizing SAE activations is an efficient substitute for feature selection — [ACL Anthology EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1521/)
- Structural critique: "Sparse Autoencoders Do Not Find Canonical Units of Analysis" (arXiv 2502.04878, 2025) — [arXiv](https://arxiv.org/pdf/2502.04878). Evaluation infrastructure is still being built: *SynthSAEBench* (Feb 2026, title only) — [arXiv 2602.14687](https://arxiv.org/pdf/2602.14687)

### Inferences
- The disagreement between Kantamneni et al. and the EMNLP 2025 paper likely comes from baseline strength: tuned logistic regression on raw residual activations vs weaker "hidden-state" baselines. This is an inference; verify before relying on it. For a production router, a raw-activation probe or DiffMean direction is the default to beat. SAEs mainly add interpretable, auditable label features, such as knowing why a message was routed to "billing".
- The cross-model transfer of SAE-feature classifiers (2B→9B within one family) is one of the few pieces of evidence that a classifier defined on interpretable features might survive a backbone change. It is directly relevant to the portability gap in Q9.

### Gaps
- Anthropic's Oct 2024 "features as classifiers" circuits update and Google DeepMind's March 2025 post deprioritising SAEs after negative downstream-probing results were not fetched in this session. Their exact numbers are not recorded here.
- No source compared SAE probes against fine-tuned LoRA classifiers on the same many-class routing task.
- The number of datasets in Kantamneni et al. was not confirmed from the fetched snippets.

## 3. Hypernetworks and generated adapters (HyperFormer, Text-to-LoRA, Doc-to-LoRA, SHINE, Drag-and-Drop, parametric skills)

### Takeaway
2025–2026 is the breakout period for "generate the adapter instead of training it". Text-to-LoRA (ICML 2025) turns a task description into a LoRA in one forward pass. Doc-to-LoRA (Feb 2026, ICML 2026) and SHINE (Feb 2026, ICML 2026) turn documents or context into LoRAs in under a second. Parametric Skills (June 2026) turns textual agent skills into LoRAs and beats ICL. The main costs are expensive meta-training (days to weeks on multiple GPUs) and a remaining quality gap versus task-trained adapters. None of these systems targets classification or routing heads specifically.

### Cited Findings
- HyperFormer (2021, context): a shared hypernetwork generates task- and layer-conditioned adapter weights for multi-task learning, the direct ancestor of this line — [Karimi Mahabadi et al., arXiv 2106.04489](https://arxiv.org/abs/2106.04489)
- **Text-to-LoRA (T2L)** (Charakorn, Cetin, Tang, Lange; Sakana AI; arXiv 2025-06-06; ICML 2025). A hypernetwork builds LoRAs from a natural-language task description in one cheap forward pass. When trained on 9 pre-trained LoRAs (GSM8K, ARC, etc.), the reconstructed LoRAs match task-specific adapters. It can compress hundreds of LoRAs and zero-shot generalize to unseen tasks — [arXiv abs](https://arxiv.org/abs/2506.06105); [ICML poster](https://icml.cc/virtual/2025/poster/43471)
- T2L details from Sakana's project page: base Mistral-7B-Instruct; generated LoRA targets q_proj and v_proj, rank 8, in all layers (~3.4M adapter params); trained on 479 tasks from the Lots-of-LoRAs collection. Zero-shot on unseen benchmarks it beats the base model and a multi-task LoRA baseline, and performance improves with hypernetwork size and data. Reconstruction training compresses existing adapters, but SFT training is what enables zero-shot generalization — [Sakana Doc-to-LoRA/Text-to-LoRA page](https://pub.sakana.ai/doc-to-lora/); [Sakana T2L page](https://sakana.ai/text-to-lora/)
- **Doc-to-LoRA (D2L)** (Sakana AI; arXiv 2602.15902, Feb 2026; ICML 2026):
  - Setup: a Perceiver-style hypernetwork (8 cross-attention blocks, ~309M params) generates rank-8 LoRAs on the MLP layers of Gemma-2-2b-it, so a document is internalized and later queries run without it in context.
  - Speed and memory: updates take under a second with ~1 GB memory, versus ~40 s for oracle context distillation and 100+ s with 40+ GB for traditional context distillation. Memory stays under 50 MB regardless of length, versus 12+ GB of KV cache for 128K tokens.
  - Quality: 83.5% of the full-context upper bound on SQuAD and ~85% relative on long-context QA. Needle-in-a-haystack is near-perfect up to 40K tokens after training only on 256-token inputs, about 5x beyond the native context window.
  - Stated limitations and future work: meta-training takes "days to weeks on multiple GPUs" and there are quality trade-offs. The team plans a "foundation hypernetwork", a shared "update API" for composable adapters, and models that "nap" to distill interactions into adapters — [Sakana project page](https://pub.sakana.ai/doc-to-lora/); [arXiv PDF](https://arxiv.org/pdf/2602.15902)
- **SHINE** (Scalable Hyper In-context NEtwork; arXiv 2602.06358, Feb 2026; ICML 2026): maps context to a LoRA in one pass by reusing the frozen LLM's own parameters inside the hypernetwork. At inference only the question is given, and the context lives in the LoRA. It "achieves parity with, or occasionally exceeds, In-Context learning" and beats Generative Adapter in most cases — [arXiv HTML](https://arxiv.org/html/2602.06358v3); [GitHub](https://github.com/Yewei-Liu/SHINE)
- **Parametric Skills** (Zhao et al.; arXiv 2606.30015, 2026-06-29): a hypernetwork converts free-form textual skills into LoRA adapters at test time for "context-free skill exploitation". It is trained on a large skill library with synthesized trajectories. It beats ICL by 6.44 points (DeepSeek-V4-Flash evaluation) across six software-engineering subtasks, and the authors frame it as a step toward "test-time continual learning" — [arXiv abs](https://arxiv.org/abs/2606.30015)
- Related 2026 title: *LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents* (June 2026, title only) — [arXiv 2606.06087](https://arxiv.org/pdf/2606.06087)
- HyperSteer (June 2025) applies the same hypernetwork idea to activation steering vectors instead of LoRA weights (see Q1) — [arXiv](https://arxiv.org/abs/2506.03292)

### Inferences
- The hypernetwork frontier has moved from task→adapter (T2L) to document→adapter (D2L, SHINE) and skill→adapter (Parametric Skills). All of them target generation or QA quality. A "label-taxonomy → classifier intervention" hypernetwork would let a router adapt instantly when Slack channels or ticket categories are added or renamed. Nothing found addresses that, and it looks like an open niche (this is inference, not a cited gap).
- The meta-training cost (days to weeks on multiple GPUs per base model) is a portability liability: a hypernetwork is tied to one base model's weight shapes. That compounds the upgrade problem in Q9.

### Gaps
- Drag-and-Drop LLMs (prompt-to-weights, June 2025) was not fetched, so no numbers are recorded for it.
- Exact T2L zero-shot numbers per benchmark were not obtained; the abstract gives only qualitative claims.
- No results were found for hypernetwork-generated adapters on classification benchmarks with large label spaces.

## 4. Memory and knowledge layers (Memory Layers at Scale, product-key memory, Cartridges/trained KV caches, knowledge modules, kNN/retrieval-style plug-ins)

### Takeaway
Three families now compete with fine-tuning for adding knowledge:
- **Sparse memory layers** (Meta, Dec 2024; sparse memory finetuning, Oct 2025) forget far less than LoRA when learning new facts.
- **Trained KV caches ("Cartridges", June 2025, ICLR 2026)** match ICL on a corpus with 38.6x less memory and 26.4x higher throughput.
- **Plug-in modules**: knowledge modules via deep context distillation (2025), and Memory Decoder (NeurIPS 2025), which transfers across any model that shares the tokenizer.

Newer 2026 work shows these precomputed memories degrade when composed and are costly to keep current.

### Cited Findings
- **Memory Layers at Scale** (Berges et al., Meta; arXiv 2412.09764, Dec 2024; ICML 2025): a trainable key-value lookup adds parameters without adding FLOPs. It scales to 128B memory parameters trained on 1T tokens, compared against base models up to 8B. Memory-augmented models beat dense models with more than 2x the compute and beat compute- and parameter-matched MoEs. Gains are largest on factual QA (NaturalQuestions, TriviaQA) — [arXiv abs](https://arxiv.org/abs/2412.09764); [Meta AI](https://ai.meta.com/research/publications/memory-layers-at-scale/)
- Scaling follow-up: *UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning* (Aug 2025, title only) — [arXiv 2508.18756](https://arxiv.org/pdf/2508.18756)
- **Continual Learning via Sparse Memory Finetuning** (Jessy Lin et al., Meta/UC Berkeley; arXiv 2510.15103, 2025-10-16):
  - Method: update only the memory slots that are highly activated by the new knowledge relative to their usage on pretraining data. Each forward pass touches about 10k parameters out of a 1–10M-slot pool.
  - Result: after learning new facts, NaturalQuestions F1 drops 89% with full finetuning and 71% with LoRA, but only 11% with sparse memory finetuning, at the same level of new-knowledge acquisition — [arXiv abs](https://arxiv.org/abs/2510.15103)
  - 2026 follow-ups (title only): *Improving Sparse Memory Finetuning* (Apr 2026) — [arXiv 2604.05248](https://arxiv.org/pdf/2604.05248); *Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning* (May 2026) — [arXiv 2605.03229](https://arxiv.org/pdf/2605.03229)
- **Cartridges** (Eyuboglu et al.; arXiv 2506.06266, June 2025; ICLR 2026):
  - Method: train a small KV cache offline per corpus using "self-study", which generates synthetic conversations about the corpus and trains the cache with a context-distillation objective. At inference the Cartridge is loaded in place of the corpus.
  - Result: matches ICL on long-context benchmarks with 38.6x less memory and 26.4x higher throughput. Prompt and KV-compression baselines degrade quickly beyond ~2x compression, while Cartridges hold up — [arXiv abs](https://arxiv.org/abs/2506.06266); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011902)
  - Follow-up, title only: *Learned Structure in Cartridges: Keys as Shareable Routers in Self-Studied Representations* (Aug 2025) — [arXiv 2508.17032](https://arxiv.org/pdf/2508.17032)
- **What It Costs to Compose, Rebuild, and Correct Precomputed Memory** (Asa Shepard; arXiv 2608.30647, 2026-08-31; Llama-3.1-8B-Instruct, saved KV caches and trained KV compressions):
  - Precomputed memory loses accuracy when assembled from separately prepared parts.
  - Keeping memories current requires rebuilds costing "a large fraction of full preparation".
  - Corrections served alongside the memory are ignored, depending on phrasing.
  - Open problems it names: rebuild cadence, cost-efficient rebuilding, and interim correction versus full rebuild. It suggests "warm-rebuilding trained compressions" as promising — [arXiv abs](https://arxiv.org/abs/2608.30647)
- **Knowledge Modules via Deep Context Distillation** (Caccia, Ansell, Ponti, Vulić, Sordoni; Microsoft Research; arXiv 2503.08727, Mar 2025; COLM 2025): document-level LoRA "knowledge modules" can be plugged in on demand. Next-token prediction is a poor objective for training them. Instead, the module is trained to match the hidden states and logits of a teacher that has the document in context — [arXiv abs](https://arxiv.org/abs/2503.08727); [Microsoft Research](https://www.microsoft.com/en-us/research/publication/training-plug-and-play-knowledge-modules-with-deep-context-distillation/)
- **Memory Decoder** (arXiv 2508.09874, Aug 2025; NeurIPS 2025): a small transformer decoder is pretrained to imitate a non-parametric kNN-style retriever's output distribution. It then plugs into any LM sharing the same tokenizer without model-specific changes. It adapted various Qwen and Llama models to biomedicine, finance and law, cutting perplexity by 6.17 points on average, while avoiding DAPT's forgetting and RAG's latency — [arXiv abs](https://arxiv.org/abs/2508.09874); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119458). Follow-up, title only: *Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory* (Jul 2026) — [arXiv 2607.27919](https://arxiv.org/pdf/2607.27919)
- Other 2026 frozen-LLM memory work (title and snippet only): *Trained Persistent Memory for Frozen Decoder-Only LLMs* (Mar 2026). It reports that persistent latent-space memory for a frozen LLM is feasible with small trainable adapters — [arXiv 2603.22329](https://arxiv.org/html/2603.22329v1)

### Inferences
- For classification and routing, these knowledge-oriented methods matter less than representation methods, with one exception: keeping a router current as organizational knowledge changes (new teams, new products). Sparse memory finetuning's low forgetting and Cartridges' per-corpus caching suggest a design with one frozen backbone, a per-tenant or per-org memory component, and a small classifier head. No source evaluated that design.
- The title of the Cartridges follow-up ("Keys as Shareable Routers") hints that learned KV prefixes develop routing structure. That could connect KV-cache adaptation to routing tasks, but the paper was not read.

### Gaps
- Product-key memory specifically for classification tasks: no sources found.
- The Memory Decoder at Scale and UltraMemV2 results were not read.

## 5. Composition and merging (task arithmetic, TIES/DARE, LoRAHub, MoLE/X-LoRA, adapter routing, modular LLMs)

### Takeaway
Composition works for a handful of skills. A 2026 survey stresses that full-model merging methods do not carry over directly to LoRA, and that when the number of skills is large, training on a data mix still beats merging. Mixture-of-LoRA-experts with learned routing (2024–2026) is the active answer. That makes the "router" part of modular LLMs itself a classification problem, which is a nice symmetry for message routing.

### Cited Findings
- Foundations (context): task arithmetic, i.e. adding and subtracting fine-tuning deltas — [Ilharco et al., arXiv 2212.04089](https://arxiv.org/abs/2212.04089); TIES merging — [Yadav et al., arXiv 2306.01708](https://arxiv.org/abs/2306.01708); DARE drop-and-rescale of delta params — [Yu et al., arXiv 2311.03099](https://arxiv.org/abs/2311.03099); LoRAHub, gradient-free few-shot composition of LoRA modules — [Huang et al., arXiv 2307.13269](https://arxiv.org/abs/2307.13269)
- 2026 survey ("Model Merging in the Era of Large Language Models", arXiv 2603.09938; the Awesome-Model-Merging list notes ACM Computing Surveys 2026):
  - LoRA-specific composition is "a particularly active subfield", because merging low-rank adapters raises questions of rank allocation and subspace compatibility, and full-model merging methods "do not transfer directly to the LoRA setting".
  - Recent LoRA-merging methods it catalogs: CoMoL (2026), MeteoRA (2025), HydraOpt (2025), Tensorized Clustered LoRA (2025), FlyLoRA rank-wise MoE (2025), IterIS (2025), Adaptive LoRA Merge (2025) — [arXiv 2603.09938](https://arxiv.org/pdf/2603.09938); [GitHub list](https://github.com/ennengyang/awesome-model-merging-methods-theories-applications)
- LoRA Soups (Oct 2024): "when the number of skills is large, DATA-MIX is still the best strategy for skill composition", so merging does not yet scale to many skills — [arXiv 2410.13025](https://arxiv.org/html/2410.13025v1)
- Mixture of LoRA Experts (MoLE; arXiv 2404.13628, Apr 2024): a hierarchical gate over multiple LoRAs — [arXiv](https://arxiv.org/abs/2404.13628)
- Routing variants:
  - LD-MoLE (Sep 2025) uses a differentiable, token- and layer-wise learnable routing function in place of TopK — [arXiv 2509.25684](https://arxiv.org/abs/2509.25684)
  - DynMoLE (Apr 2025) uses hybrid routing — [arXiv 2504.00661](https://arxiv.org/pdf/2504.00661)
  - Retrieval-augmented MoLE for "uploadable ML" (2024) — [arXiv 2406.16989](https://arxiv.org/pdf/2406.16989)
  - *Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs* (June 2026, title only) — [arXiv 2606.31413](https://arxiv.org/html/2606.31413v1)
  - *LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging* (ACL 2026) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1837.pdf)
- Interference-focused merging: *Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging* (May 2025) — [arXiv 2505.22934](https://arxiv.org/pdf/2505.22934); KnOTS, SVD-aligned LoRA merging (Oct 2024) — [arXiv 2410.19735](https://arxiv.org/pdf/2410.19735)
- LoRAtorio (Aug 2025, title only): "An intrinsic approach to LoRA skill composition" — [arXiv 2508.11624](https://arxiv.org/pdf/2508.11624)

### Inferences
- For a multi-task shared backbone serving many classifiers, merging is the wrong tool: each classifier needs its own decision boundary. Instance-level routing or selection of modules, or running many lightweight heads or interventions in parallel on a shared forward pass, fits better. The MoLE router is itself a learned classifier over activations, so the machinery built for message routing and for adapter routing is the same.
- Scaling to many skills is an explicitly acknowledged failure mode (LoRA Soups). An enterprise with dozens of routing taxonomies sits squarely in that regime.

### Gaps
- No quantitative numbers from X-LoRA or MoLE were extracted in this session.
- No study was found on composing many classification-only modules, as opposed to generative skills.

## 6. Prompt-based parameter additions (prefix tuning, P-tuning v2, soft prompts, gist tokens, prompt compression): latest results

### Takeaway
Soft-prompt families are mature and largely settled. They are competitive at scale and uniquely convenient for batching many tasks on one frozen model, but they get less research attention in 2025–2026 than LoRA, hypernetworks and trained KV caches. The live frontier is prompt/context compression (gist tokens, 500xCompressor, Cartridges). Cartridges (Q4) can be read as a trained prefix that finally beats compression baselines at high ratios.

### Cited Findings
- Prefix tuning (2021): trainable continuous prefixes at every layer — [Li & Liang, arXiv 2101.00190](https://arxiv.org/abs/2101.00190). Prompt tuning: the gap to full fine-tuning closes as model scale increases — [Lester et al., arXiv 2104.08691](https://arxiv.org/abs/2104.08691). P-tuning v2: deep prompt tuning comparable to fine-tuning across scales and NLU tasks — [Liu et al., arXiv 2110.07602](https://arxiv.org/abs/2110.07602)
- Gist tokens: the attention mask is modified so a few learned "gist" tokens stand in for the prompt, with caching of their activations. Compression reaches up to 26x — [Prompt Compression survey, arXiv 2410.12388](https://arxiv.org/pdf/2410.12388); original: [Mu, Li, Goodman, arXiv 2304.08467](https://arxiv.org/abs/2304.08467)
- The Prompt Compression survey (Li, Liu, Su, Collier; NAACL 2025 main, selected oral) splits the field into hard-prompt methods (removal, summarization, restructuring) and soft-prompt methods (LLM-fixed and gist-token-based) — [GitHub](https://github.com/ZongqianLi/Prompt-Compression-Survey); [arXiv 2410.12388](https://arxiv.org/pdf/2410.12388)
- 500xCompressor (Aug 2024): generalized prompt compression to extreme ratios — [arXiv 2408.03094](https://arxiv.org/pdf/2408.03094)
- Cartridges' comparison: prompt- and KV-compression baselines degrade quickly beyond ~2x compression, while self-study-trained KV caches keep ICL-level quality at 38.6x memory reduction — [Cartridges arXiv](https://arxiv.org/abs/2506.06266)

### Inferences
- For classification on one frozen backbone, a per-task soft prompt or trained KV prefix is a cheap, batchable per-task artifact. It is architecturally identical to a Cartridge, just trained on labels rather than by self-study. The newer training objectives (context distillation, self-study) have not, in the sources found, been applied to classification prefixes.

### Gaps
- No 2025–2026 head-to-head of soft prompts vs LoRA vs ReFT on large-label-space classification with modern 7B+ decoder LLMs was found.

## 7. Tiny-parameter methods (BitFit, IA3, LayerNorm tuning, VeRA, LoRA-XS, TinyLoRA, rank-1): how few parameters suffice?

### Takeaway
Remarkably few. In 2026, TinyLoRA trained Qwen2.5-8B to 91% GSM8K with 13 trainable parameters (26 bytes), but only with RL; SFT needed 100–1000x larger updates. "LoRA Without Regret" (Thinking Machines, Sept 2025) showed rank-1 LoRA matches full fine-tuning for RL. For classification, LoRA-XS and VeRA reach LoRA-level GLUE accuracy with orders of magnitude fewer parameters, and LoReFT does so at 0.015% of parameters.

### Cited Findings
- **TinyLoRA / "Learning to Reason in 13 Parameters"** (arXiv 2602.04118, Feb 2026; now in the Hugging Face PEFT library):
  - Mechanism: builds on LoRA-XS (an SVD of the frozen weights) and projects a tiny trainable vector through fixed random tensors.
  - Result: Qwen2.5-8B reaches 91% on GSM8K with 13 bf16 parameters (26 bytes). On AIME, AMC and MATH500 it recovers 90% of the improvement while training 1000x fewer parameters.
  - Caveat: strong results require RL. SFT-trained models need 100–1000x larger updates for the same performance. The hypothesis is that sparse, clean reward signals carry little information — [arXiv abs](https://arxiv.org/abs/2602.04118v1); [HF PEFT docs](https://huggingface.co/docs/peft/en/package_reference/tinylora)
- **LoRA Without Regret** (John Schulman and Thinking Machines Lab; blog, Sept 2025): LoRA matches full fine-tuning for RL even at rank 1, because policy gradients absorb only about 1 bit per episode, whereas SFT delivers O(tokens) bits. It recommends applying LoRA to all weight matrices, with learning rates about 10x those of full FT (~1e-4 to 5e-4) — [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/)
- **LoRA-XS** (arXiv 2405.17604, 2024): inserts a small trainable r×r matrix between frozen SVD-derived subspaces of the pretrained weights. The trainable count can go from one parameter per module upward. On six GLUE tasks with RoBERTa-large, rank-16 LoRA-XS beats VeRA with 2.5x fewer trainable parameters, dropping only ~4 points at rank 4. It cuts parameters by more than 100x versus LoRA on 7B models — [arXiv PDF](https://arxiv.org/pdf/2405.17604v2); [GitHub](https://github.com/mohammadrezabanaei/lora-xs)
- **VeRA** (Kopiczko et al.; arXiv 2310.11454, Oct 2023; ICLR 2024): one pair of frozen random low-rank matrices is shared across all layers, and only small per-layer scaling vectors are trained — [arXiv](https://arxiv.org/abs/2310.11454); [HF PEFT](https://huggingface.co/docs/peft/en/package_reference/vera)
- Newer extreme-PEFT titles: *Uni-LoRA: One Vector is All You Need* (June 2025) — [arXiv 2506.00799](https://arxiv.org/pdf/2506.00799); VB-LoRA, vector banks (NeurIPS 2024) — [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e0d38c676d5855bcfab7f6d29d20ad9-Paper-Conference.pdf)
- Classic tiny methods (context): BitFit trains only biases — [Ben Zaken et al., arXiv 2106.10199](https://arxiv.org/abs/2106.10199). IA3 learns vectors that rescale keys, values and FFN activations; its T-Few recipe showed few-shot PEFT beating ICL — [Liu et al., arXiv 2205.05638](https://arxiv.org/abs/2205.05638). LayerNorm-only tuning — [Zhao et al., arXiv 2312.11420](https://arxiv.org/abs/2312.11420). Intrinsic dimensionality: very low-dimensional reparameterizations recover most fine-tuning performance on NLU — [Aghajanyan et al., arXiv 2012.13255](https://arxiv.org/abs/2012.13255)
- ReFT: LoReFT matches LoRA on GLUE at 0.014–0.015% of parameters (see Q1) — [ReFT HTML](https://arxiv.org/html/2404.03592v3)

### Inferences
- Classification is information-light: a label carries log2(K) bits per example. By TinyLoRA's and "LoRA Without Regret"'s information-theoretic argument, it should need very few adapted parameters. Nobody seems to have tested whether RL-with-correctness-reward on a tiny adapter (or on a rank-1 ReFT) beats SFT or cross-entropy for a classification/routing head at matched parameter budget. That looks like a small, cheap, genuinely novel experiment (inference).
- Kilobyte-scale adapters would make per-customer or per-channel routers essentially free to store and hot-swap on one backbone. This is context for the economics researcher.

### Gaps
- TinyLoRA results cover math reasoning only. No classification results at the 10–1000 parameter scale on modern decoder LLMs were found.
- Exact VeRA and LoRA-XS GLUE numbers per task were not extracted.

## 8. Test-time training / test-time adaptation (2024–2026) and self-adapting LLMs (SEAL)

### Takeaway
Test-time training (TTT) has gone from a niche idea to a real capability:
- Per-instance LoRA TTT reached 53% on ARC with an 8B model (Nov 2024, ICML 2025).
- SEAL (MIT, June 2025) learns via RL to generate its own fine-tuning data and directives.
- TTT-E2E (Stanford/NVIDIA, Dec 2025) treats long context as test-time learning into the weights, with constant latency.

The main limits are forgetting, high per-update cost, and the need for task-paired rewards.

### Cited Findings
- **Test-time training for few-shot learning** (Akyürek et al.; arXiv 2411.07279, Nov 2024; ICML 2025 under the title "...for Few-Shot Learning"): temporary per-instance parameter updates from a loss on in-context examples give up to 6x higher accuracy than fine-tuned baselines on ARC. An 8B LM reaches 53.0% on the public validation set, and 61.9% when ensembled with program synthesis, matching average human performance. The key ingredients are initial fine-tuning on similar tasks, auxiliary task formats and augmentations, and per-instance training — [arXiv abs](https://arxiv.org/abs/2411.07279); [PMLR](https://proceedings.mlr.press/v267/akyurek25a.html)
- **SEAL: Self-Adapting Language Models** (Zweiger, Pari, Guo, Akyürek, Kim, Agrawal; arXiv 2506.10943, v1 2025-06-12, v2 2025-09-18):
  - Mechanism: the model writes "self-edits" (restructured training data, optimization hyperparameters, tool calls). These are applied as persistent weight updates via SFT, and an RL outer loop rewards the self-edits by downstream performance of the updated model — [arXiv abs](https://arxiv.org/abs/2506.10943)
  - Knowledge incorporation (SQuAD without context, Qwen2.5-7B): base 32.7%; train on passage 33.5%; passage plus synthetic data 39.7%; passage plus GPT-4.1 synthetic data 46.3%; SEAL 47.0%.
  - Few-shot ARC subset (Llama-3.2-1B-Instruct): ICL 0%; TTT with self-edits but no RL 20%; SEAL 72.5%; oracle TTT 100%.
  - Limitations: each self-edit evaluation takes ~30–45 s; performance on earlier tasks gradually declines as edits accumulate (catastrophic forgetting); every context needs paired downstream evaluation tasks — [arXiv HTML v2](https://arxiv.org/html/2506.10943v2)
- **End-to-End Test-Time Training for Long Context (TTT-E2E)** (Stanford and NVIDIA; arXiv 2512.23675, Dec 2025): long-context modeling is reframed as continual learning. A standard transformer with sliding-window attention keeps learning at test time via next-token prediction on the context, compressing it into the weights, and meta-learning improves the initialization. At 3B parameters and 164B training tokens it scales with context length like full attention (Mamba 2 and Gated DeltaNet do not) and is 2.7x faster than full attention at 128K, with constant latency — [arXiv abs](https://arxiv.org/abs/2512.23675); [TechTalks coverage, Jan 2026](https://bdtechtalks.com/2026/01/12/nvidia-end-to-end-test-time-training/)
- 2026 titles (title only): *Test-Time Training with Next-Token Prediction* (June 2026) — [arXiv 2606.21803](https://arxiv.org/pdf/2606.21803); *Self-Guided Test-Time Training for Long-Context LLMs* (July 2026) — [arXiv 2607.09415](https://arxiv.org/pdf/2607.09415)
- Earlier TTT: test-time training on retrieved nearest neighbors, where a model is fine-tuned briefly on neighbors of each test input — [Hardt & Sun, arXiv 2305.18466](https://arxiv.org/abs/2305.18466)
- Hypernetwork approaches (D2L, SHINE, Parametric Skills; Q3) are effectively amortized TTT: one forward pass replaces gradient steps at test time. Sakana explicitly lists models that "nap" to distill interactions into adapters as future work — [Sakana D2L page](https://pub.sakana.ai/doc-to-lora/)

### Inferences
- For routing, TTT on retrieved nearest-neighbor labeled messages (Hardt & Sun style) applied to a tiny adapter or intervention could handle drift in message distributions without retraining. No source tested this for classification (inference).
- SEAL's forgetting and per-edit cost make it impractical for high-throughput classification today. Its RL-over-self-generated-data idea could apply to generating augmentation data for new routing categories.

### Gaps
- No 2025–2026 TTT results specific to text classification or routing were found.
- Whether SEAL was published at a peer-reviewed venue was not confirmed in this session.

## 9. Appending layers/blocks to a frozen LLM, and plug-in modules that transfer across base model versions (adapter portability)

### Takeaway
Portability is the most practically important and least solved problem. UpgradeBench (Aug 2026), on a 14-month Qwen lineage, gives four findings that bear directly on routing:
1. Frozen intent-classification specialists keep 99–101% of their attainable gains across nine upgrade hops, while text-to-SQL specialists lose up to 59% on a single hop.
2. Naively copying an adapter to an architecturally identical but independently pretrained base fails catastrophically (Banking77: 92.8% → 42.9%).
3. Copying works across continued-pretraining releases.
4. Label-free "refresh" distillation from the old specialist reaches parity.

Transfer methods such as Trans-LoRA (2024), LoRA-X (ICLR 2025), LoRASuite (NeurIPS 2025) and Cross-LoRA (2025) exist. Memory Decoder is portable across any model sharing a tokenizer.

### Cited Findings
- **UpgradeBench** (Ye Chen, Weining Zhang; Alibaba Group / CKGSB; arXiv 2608.20918, 2026-08-21). Setup: it measures the "base-model upgrade tax" over Qwen 1.5 → 2 → 2.5 → 3 at 7–8B and 1.5–1.8B, validated on OLMo checkpoints. Strategies compared: Freeze; Copy/PortDiff; Refresh-D (the old specialist labels retained inputs and a new adapter is trained on the new base); and Retrain (QLoRA with gold labels). Results:
  - **Durability depends on "competence provenance":** intent-classification specialists (a data-bound label taxonomy) retain 99–101% of attainable gains across nine hops, while text-to-SQL (a base-bound skill) forfeits up to 59% on a single hop.
  - **Copying depends on training continuity, not architecture:** copying Qwen2 → Qwen2.5 adapters collapses Banking77 from 92.8% to 42.9%. Copying to the Qwen2.5-1M continued-pretraining release retains 0.82–1.45. On OLMo, retention is 0.88–0.99 after 46B tokens of continued pretraining but floor-level after 2.9T tokens.
  - **Refresh without gold labels:** Refresh-D reaches retention of 0.96–1.05 at 39–331 GPU-minutes, often costing more compute than gold-label retraining.
  - **Decision policy:** a prospectively specified policy gets 0.37pp mean regret over 33 episodes at one third of the compute and label budget of always-retrain. The naive shape-compatible-copy heuristic has 17.11pp regret.
  - **Small evidence sets:** at n ≤ 300, a three-zone gate cuts false-positive upgrade decisions from 13–18% to ~0.2%.
  - **Stated gaps:** no new transfer technique is proposed. Learned parameter mappings are unmeasured on shape-incompatible hops. All results use rank-16 LoRA; at rank 64 the Banking77 copy collapse weakens (R = +0.29 vs −0.55). The decay curve rests on only two endpoints. Data-free synthetic-input refresh is unmeasured. Specialists are capped at 8B — [arXiv HTML](https://arxiv.org/html/2608.20918)
- **Trans-LoRA** (arXiv 2405.17258, May 2024): data-free transfer of LoRAs across base-model upgrades. Synthetic data is generated and filtered by discriminators trained on a handful of seed inputs, then the LoRA is re-distilled on the new base. It matches or beats the source LoRA in every tested setting — [arXiv PDF](https://arxiv.org/pdf/2405.17258)
- **LoRA-X** (arXiv 2501.16559, Jan 2025; ICLR 2025): training-free, data-free LoRA transfer that keeps the adapter within the target's column/row subspace. It was demonstrated on text-to-image models (SD v1.5, SDXL) — [arXiv abs](https://arxiv.org/abs/2501.16559)
- **LoRASuite** (arXiv 2505.13515, May 2025; NeurIPS 2025): catalogs six kinds of upgrade mismatch: vocabulary, hidden size, FFN intermediate size, depth, head count, and attention type. It computes transfer matrices from the old and new weights, maps layers and heads with CKA and cosine similarity, and then does a small fine-tune. On MiniCPM and Qwen it beats full LoRA retraining by +1.4 and +6.6 points on math, using 5.5 GB less memory and 78.23% less time — [arXiv HTML](https://arxiv.org/html/2505.13515v1); [NeurIPS 2025 anthology](https://mlanthology.org/neurips/2025/li2025neurips-lorasuite/)
- **Cross-LoRA** (arXiv 2508.05232, Aug 2025): data-free and training-free LoRA transfer across heterogeneous LLMs. LoRA-Align does rank-truncated SVD subspace alignment with a Frobenius-optimal linear map, and LoRA-Shift projects the source update into the target space — [arXiv PDF](https://arxiv.org/pdf/2508.05232)
- **Memory Decoder** (NeurIPS 2025) plugs into "any pretrained language model that shares the same tokenizer" without modification — [arXiv abs](https://arxiv.org/abs/2508.09874)
- SAE-feature classifiers transferred from Gemma 2 2B to 9B-IT (EMNLP 2025) — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1521/)
- **Activated LoRA (aLoRA)** (Greenewald et al., IBM Research; arXiv 2504.12397, v1 2025-04-16, v5 2025-10-02): the adapter only modifies tokens after it is invoked, so it reuses the base model's existing KV cache. Specialized "intrinsics" (targeted checks or classifiers inside a conversation) can then be switched on mid-chain without recomputing the prefix. Accuracy is competitive with standard LoRA, and the code is in Hugging Face PEFT — [arXiv abs](https://arxiv.org/abs/2504.12397)
- Other appended-module work (context; other researchers cover LLaMA Pro and side networks): *X-Fusion: Introducing New Modality to Frozen Large Language Models* (2025) — [Pith review](https://pith.science/paper/2504.20996); *TFGN: Task-Free, Replay-Free Continual Pre-Training Without Catastrophic Forgetting at LLM Scale* (May 2026, title only) — [arXiv 2605.15053](https://arxiv.org/pdf/2605.15053); *Trained Persistent Memory for Frozen Decoder-Only LLMs* (Mar 2026) — [arXiv 2603.22329](https://arxiv.org/html/2603.22329v1)

### Inferences
- The UpgradeBench result is the single most relevant finding for a Slack/ticket router. Label-taxonomy classifiers are "data-bound", so a frozen old specialist stays good, and distillation-refresh onto a new base with unlabeled retained traffic reaches parity. The collapse on naive copy shows that the adapter itself is not a portable artifact; the portable asset is the labeled or old-model-labeled data. That supports storing routers as small data-plus-teacher pipelines rather than as weights.
- A research opening follows. Interventions defined in a model-agnostic space could be portable across base upgrades without re-distillation. Candidates include text-anchored directions, SAE features matched across models, relative/anchor representations, or tokenizer-level plug-ins like Memory Decoder. UpgradeBench explicitly leaves learned mappings on shape-incompatible hops unmeasured, and no source tests ReFT, steering-vector or probe portability across independently pretrained releases.
- aLoRA-style "activate mid-sequence, reuse the prefix KV" is the right serving primitive for many classifiers on one backbone: one shared prefill of a Slack message, then N cheap classification heads or adapters. That is complementary to representation interventions.

### Gaps
- No 2025–2026 study was found that directly appends new transformer blocks to a frozen LLM for classification tasks, beyond the architectures covered by other researchers.
- Portability of representation interventions (ReFT, steering, I2CL vectors) across model versions: no sources found.
- aLoRA speedup numbers were not in the abstract.

## 10. Explicit open problems and research gaps a small team could contribute to (especially classification/routing and a multi-task shared frozen backbone)

### Takeaway
Explicitly stated open problems cluster around five themes:
- (a) portability across base-model upgrades (UpgradeBench, Aug 2026);
- (b) scaling composition to many skills (LoRA Soups; the 2026 merging survey);
- (c) cost and staleness of precomputed memory (Aug 2026);
- (d) the reliability of steering and SAEs and rigorous evaluation against strong baselines (2025);
- (e) expensive meta-training and forgetting for hypernetworks and self-adapting models (2025–2026).

The most tractable novel contributions for a small team sit where these meet classification and routing on one frozen backbone.

### Cited Findings
- **Portability**: learned parameter mappings on shape-incompatible upgrade hops are unmeasured, as are data-free synthetic-input refresh, rank dependence of copy collapse, and fuller decay curves over continued-pretraining distance — [UpgradeBench](https://arxiv.org/html/2608.20918). The underlying motivation is that LoRA adapters "cannot be directly applied to another [model] without retraining", which is especially problematic when bases are deprecated — [LoRA-X ICLR 2025](https://iclr.cc/virtual/2025/poster/30873); [Cross-LoRA](https://arxiv.org/pdf/2508.05232)
- **Composition at scale**: "when the number of skills is large, DATA-MIX is still the best strategy... improving current model merging schemes is an important future direction" — [LoRA Soups](https://arxiv.org/html/2410.13025v1); full-model merging methods do not transfer directly to LoRA — [Model merging survey 2026](https://arxiv.org/pdf/2603.09938)
- **Memory maintenance**: open problems include cost-efficient rebuilding, rebuild cadence, and interim correction versus full rebuild; precomputed memories degrade when composed — [Shepard 2026](https://arxiv.org/abs/2608.30647)
- **Evaluation rigor**: interpretability methods need to be "rigorously evaluate[d]... on downstream tasks with strong baselines" — [Kantamneni et al. 2025](https://arxiv.org/pdf/2502.16681). Steering is brittle out of distribution and influenced by spurious biases — [Tan et al.](https://openreview.net/forum?id=v8X70gTodR). Prompting beats representation steering on AxBench — [AxBench](https://www.alphaxiv.org/abs/2501.17148)
- **Hypernetwork cost and generality**: meta-training takes "days to weeks on multiple GPUs". Future work named: a "foundation hypernetwork", a shared "update API" for composable adapters, and "napping" memory — [Sakana D2L](https://pub.sakana.ai/doc-to-lora/)
- **Self-adaptation**: catastrophic forgetting as edits accumulate, 30–45 s per self-edit evaluation, and dependence on paired downstream tasks — [SEAL v2](https://arxiv.org/html/2506.10943v2)
- **Forgetting-free knowledge addition**: sparse memory finetuning drops NQ F1 by 11%, versus 71% for LoRA and 89% for full FT, which motivates sparse and modular updates — [Lin et al. 2025](https://arxiv.org/abs/2510.15103)
- **RL vs SFT parameter needs**: RL needs 100–1000x smaller updates than SFT for the same gain — [TinyLoRA](https://arxiv.org/abs/2602.04118v1); rank-1 LoRA suffices for RL — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- **Steering as a paradigm**: the ACL 2026 position paper calls for integrating steering into a unified adaptation taxonomy with functional criteria — [arXiv 2604.14090](https://arxiv.org/abs/2604.14090)

### Inferences
Candidate novel contributions below are the researcher's synthesis, not claims from the sources. Each is tied to the stated gaps above.

1. **Portable routers across base upgrades.** Build a benchmark and method for transferring classification interventions (LoReFT rank-1, DiffMean directions, probes, I2CL vectors) across independently pretrained releases (e.g., Qwen2 → 2.5 → 3). Compare against UpgradeBench's Freeze, Copy and Refresh-D baselines, using Banking77, CLINC150 and a Slack-like routing set. Candidate mechanisms: anchor/relative representations, CKA-based layer matching (as in LoRASuite) applied to interventions, or text-anchored label prototypes that are re-encoded on the new base. UpgradeBench's reported copy collapse (92.8% → 42.9%) and its explicitly unmeasured "learned mappings" gap make this a well-posed target.
2. **Taxonomy-to-intervention hypernetworks.** A T2L/HyperSteer-style hypernetwork that takes a label-taxonomy description (channel names, category definitions, a few examples) and emits a classification intervention or head in one pass. That would give zero-shot routers when categories change. Existing hypernetworks target generation and QA, and HyperSteer shows the approach scales to thousands of concepts.
3. **One pass, many heads.** Benchmark N classifiers served on one frozen forward pass: shared prefill, then per-task position-local ReFT interventions, aLoRA-style late activation, or probes at different layers. Measure interference, throughput and accuracy against per-task LoRA. It addresses "many skills" composition without merging.
4. **RL-trained tiny classifiers.** Test the TinyLoRA and "LoRA Without Regret" information argument on classification. For a K-way router, compare cross-entropy SFT with RL on a correctness reward, for 10–1000-parameter adapters and rank-1 ReFT, measuring sample efficiency, calibration and robustness to label noise.
5. **Low-forgetting taxonomy evolution.** Add new routes or categories via sparse memory slots or I2CL/ELICIT-style vector libraries, and measure forgetting of old routes against LoRA. Include rebuild-cadence and correction questions borrowed from the precomputed-memory work.
6. **Rigorous baselines for interpretable routers.** Test SAE-feature routers against raw-activation probes and fine-tuned LoRA under the noisy-label, imbalance and drift conditions typical of Slack and ticket data. The value would be auditability ("why was this routed?"), in line with Kantamneni et al.'s call for strong baselines.

Landscape summary (synthesized from sections 1–9):

| Family | Base weights touched? | Typical trainable size | Classification evidence | Key 2025–2026 frontier item |
|---|---|---|---|---|
| ReFT / steering / task vectors | No (activations) | ~0.01–0.03% params; rank-1 directions | GLUE parity with LoRA; ~0.94 AUROC detection | HyperSteer (Jun 2025); Steer2Adapt (Feb 2026) |
| SAE / probe classifiers | No | Linear head on features | Mixed; SAEs rarely beat raw-activation probes | SAE classifiers with cross-model transfer (EMNLP 2025) |
| Hypernetwork-generated LoRA | No (adds generated LoRA) | Hypernet ~309M; LoRA ~3.4M | None found for classification | D2L, SHINE (ICML 2026); Parametric Skills (Jun 2026) |
| Memory layers / Cartridges / Memory Decoder | No (adds memory or KV) | 10k active of 1–10M slots; small KV | Knowledge tasks, not classification | Sparse memory FT (Oct 2025); Cartridges (ICLR 2026) |
| Merging / MoLE routing | No (composes adapters) | Adapters plus router | Router is itself a classifier | 2026 survey; hard-routed LoRA mixtures (Jun 2026) |
| Soft prompts / gist / prefix | No | Few tokens | Mature; competitive at scale | Compression beyond 2x mostly fails except Cartridges |
| Tiny PEFT (TinyLoRA, LoRA-XS, VeRA) | No (adapter) | 13 params to ~kB | GLUE parity (LoRA-XS/VeRA) | TinyLoRA 13 params, RL only (Feb 2026) |
| TTT / SEAL | Temporarily or persistently via adapters | Per-instance LoRA | None for classification | TTT-E2E (Dec 2025); SEAL (Jun 2025) |
| Portability methods | No | — | Intent classification is durable; naive copy fails | UpgradeBench (Aug 2026) |

### Gaps
- No single survey was found that lists open problems across all of these families together. The gaps above are assembled from the individual papers' limitations sections and from surveys of merging, prompt compression and representation engineering.
- The recent PEFT survey literature (2025–2026) was not fetched directly, so its explicit open-problem lists are not recorded here.
- Claims about novelty (items 1–6) reflect only the sources searched in this session (Sept 2026). A targeted literature check is advised before committing to any of them. Portability of representation interventions and taxonomy-to-intervention hypernetworks in particular could have very recent work not surfaced by these searches.
