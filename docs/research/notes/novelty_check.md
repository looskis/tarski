# Novelty check: tarski (frozen shared trunk + per-task branches at split depth k)

Checked 2026-09-27. Sources: arXiv, ACL Anthology, USENIX, PMLR/NeurIPS/ICML proceedings, Semantic Scholar (the citation graph of the key paper), web search, GitHub (`gh search repos` and READMEs). This builds on the sibling notes in `Read once decision heads novelty/` and `Appending layers to language models/`; findings already verified there (DeFormer, PreTTR, MORES, LUMEN, MICE, GLiGuard, SCX Router, LEC, Layer-by-Layer) are reused, not re-fetched.

Quoting policy: sources are paraphrased with pinpoint locations (section, table or figure). The file contains one short verbatim quote. All numbers come from the primary source unless marked.

## The system under test

- **Trunk:** a frozen ModernBERT-base trunk runs once per message, up to the deepest split depth any requested decision needs.
- **Branches:** each decision has its own branch at depth k. A branch is one of two kinds:
  - a probe (norm, mean pool, linear);
  - d layers copied from the base's layers [k, k+d), fine-tuned, then pool and linear. Layers above k+d are dropped ("truncate and fork").
- **Training:** each branch is trained on its own, by end users, on cached trunk activations. The base never changes, so tasks cannot interfere.
- **Serving:** branches (KB to 20 MB) are hot-loaded and evicted with an LRU, on consumer hardware, behind a Jev-compatible API.

## Summary of verdicts

| # | Claim | Verdict | Closest prior work |
|---|---|---|---|
| 1 | Share lower-layer compute across tasks: each task gets its own upper layers on a frozen pretrained transformer, split depth varies by task, branches copied from the base's next layers | **Already done** | Wei, Qi & He, "A Flexible Multi-Task Model for BERT Serving", ACL 2022. Jiang et al., "Mainstream", USENIX ATC 2018 (vision). AdapterDrop, EMNLP 2021 (uniform depth, adapters). |
| 2 | Automatic per-task split depth (easy decisions branch early), heterogeneous-depth branches served from one trunk pass | **Partially done.** Automatic per-task depth selection and heterogeneous-depth one-pass serving exist. Using layer-wise probe accuracy on cached activations as the selection signal was not found. | Wei et al. 2022 (validation sweep plus a marginal-benefit threshold). Mainstream 2018 (deploy-time scheduler). LEC 2024 / Layer-by-Layer 2025 (probe-chosen layer, single task). |
| 3 | Independently and incrementally trained branches on a frozen shared backbone, for many-tenant local serving | **Partially done.** Every ingredient has precedent: independent per-task training with modular updates, heads trained on cached features, multi-tenant stem sharing, adapter hot-swapping, many probes on one forward pass. The combination was not found: an end-user-trained, locally served, LRU hot-loaded set of forked-layer branches whose trunk is truncated per request. | Wei et al. 2022. Mainstream 2018. Tesla HydraNet (talks). Embedding Recycling, EACL Findings 2023. Anthropic "cheap monitors" 2025. Apple adapters 2024. |
| 4 | Latency/cost of N decisions per message: shared trunk vs N separate models vs question-in-input cross-encoders, on consumer hardware | **Partially done.** Shared-trunk vs N-separate-model cost has been measured on datacenter GPUs. The three-way comparison against question-in-input decision models on Apple silicon or CPU: no prior work found. | AdapterDrop (N fully fine-tuned models vs shared layers, V100/Titan X). Wei et al. 2022 (21 tasks on V100). Laya / laya-mlx (per-question scaling, T4 and M3 Max). |
| 5 | Initialise the branch from copies of the base's own next layers ("truncate and fork") vs random-init head layers, trunk frozen | **Already done as a technique.** A systematic copy-vs-random ablation under a frozen trunk with truncation above k+d: **not found** (partial evidence only). | Wei et al. 2022 (branch layers are the base's own top layers; pretrained vs teacher init in Fig. 2). MORES 2020 (random vs BERT-copied init: large loss). LUMEN 2023. Anthropic 2025 (retrain final layers). EE-Tuning 2024. |

**Bottom line:** the architecture is prior art. The closest match is Wei et al., ACL 2022. It independently partially fine-tunes the top layers of a frozen BERT per task, picks a per-task split depth, distils the task layers into a few layers initialised from the teacher's layers just above the split, and merges everything into one graph that shares the frozen bottom. Its Xiaomi production deployment served 21 tasks from different teams. The vision systems analogue, Mainstream (2018), is also multi-tenant with independently trained per-app branch points.

A defensible thesis contribution is narrower. Its parts:
1. a cheap probe-guided split-depth selector, replacing Wei et al.'s ~10× fine-tuning sweep;
2. plain truncate-and-fork trained on cached activations, with no teacher or KD stage, plus a copy-vs-random-vs-probe ablation;
3. per-request trunk truncation with LRU hot-loading for end-user-trained branches on consumer hardware;
4. a three-way cost/accuracy comparison against Laya/Jev-style question-in-input decision models on Apple silicon and CPU;
5. a re-examination on ModernBERT, a 2024 encoder with alternating local/global attention, where layer behaviour may differ from BERT. MICE (2026) found larger losses from splitting ModernBERT-family encoders.

---

## Claim 1: Share lower-layer computation across tasks; task-specific upper layers on a frozen pretrained transformer

### Closest prior work

**Wei, Qi & He (Xiaomi XiaoAI), "A Flexible Multi-Task Model for BERT Serving", ACL 2022 (short), pp. 785–796.** [ACL Anthology](https://aclanthology.org/2022.acl-short.89/) · [arXiv 2107.05377](https://arxiv.org/abs/2107.05377) · code: [DandyQi/CentraBert](https://github.com/DandyQi/CentraBert). I read the full text.
- **Method, Sec. 3.1–3.3.**
  - Abstract, verbatim: "only fine-tune some top layers of BERT while keep the other layers frozen".
  - Each task gets its own independent copy of BERT-base, with the bottom 12−L(τ) layers frozen and the top L(τ) layers fine-tuned. The task layers are therefore the base's own layers, initialised from pretrained BERT.
  - L(τ) varies per task. It is chosen by validation over N_min=4 to N_max=10.
  - Each teacher is then distilled into a student with l(τ) task-specific layers, initialised from the bottommost N_s layers of the teacher (Sec. 3.2).
  - The students are merged into one computation graph that shares the frozen layers (Sec. 3.3).
- **Heterogeneous depths (Table 2).** Final configurations are (frozen, task-specific) pairs, for example QQP (2,1), RTE (7,5), MNLI (4,3) and CoLA (4,8). They are served as "7 + 26" layers: seven shared frozen layers, with tasks branching at depths 2, 4, 6 and 7.
- **Accuracy and cost.** GLUE average is 82.5 vs 82.8 for full fine-tuning, at 34.3% of the layer overhead.
- **Motivation (Sec. 1, Appendix E).** The paper explicitly targets modular, incremental development: updating one task must not affect the others. It shows MT-DNN cannot absorb a new task without retraining (the leave-one-out row in Table 2).
- **Production, Appendix E.** The Xiaomi XiaoAI utterance-understanding system serves 21 tasks owned by different teams:
  - 40 transformer layers in total, 8 of them shared and frozen, about 1.5 task layers per task;
  - 5 V100s, P99 32 ms at 4000 QPS;
  - compared with 42 GPUs for single-task serving, an estimated 88% cost reduction.

**Jiang et al., "Mainstream: Dynamic Stem-Sharing for Multi-Tenant Video Processing", USENIX ATC 2018.** [paper](https://www.usenix.org/conference/atc18/presentation/jiang). I read the full text.
- Applications are fine-tuned from a common pretrained CNN. Layers up to a "branchpoint" are frozen, and the layers above are specialised: fine-tuned copies of the base's own layers. M-Trainer produces one model per branchpoint (Sec. 5).
- The runtime is a tree of apps splitting from a shared stem at *different* branch points. The stem is executed once per frame (Sec. 4, 6).
- Apps are trained independently by different developers and packaged as "M-Packages" (Fig. 1).
- Reported result: up to 47% higher event F1 than "Max-Sharing" and 87× relative to "No-Sharing".

**Rücklé et al., "AdapterDrop: On the Efficiency of Adapters in Transformers", EMNLP 2021.** [ACL Anthology](https://aclanthology.org/2021.emnlp-main.626/) · [arXiv 2010.11918](https://arxiv.org/abs/2010.11918)
- Dropping adapters from the lowest n layers lets several independent classifications of the same input share those layers' computation. The frozen base plus independently trained adapters are the setting.
- Speedup per shared layer (Table 2): 4.3% for 2 tasks, 6.6% for 4, 7.8% for 8, 8.4% for 16.
- Dropping 5 layers keeps 97.1% of performance (specialized variant).
- **Gaps relative to tarski:** a single uniform drop depth for all tasks, and task-specific parts are adapters, not copies of base layers.

### Also relevant
- **Encoder splitting, IR/QA** (details in the sibling notes): DeFormer (Cao et al., ACL 2020), PreTTR (MacAvaney et al., SIGIR 2020), MORES (Gao, Dai, Callan, EMNLP 2020), LUMEN (de Jong et al., ICML 2023). The upper layers are the base's own, split at a chosen depth. There is one task per model, not many tasks sharing a trunk.
- **Joint multi-task training** (the trunk is trained jointly, not frozen):
  - MT-DNN (Liu et al., ACL 2019): hard sharing of the whole encoder.
  - Branched Multi-Task Networks (Vandenhende et al., BMVC 2020, [arXiv 1904.02920](https://arxiv.org/abs/1904.02920)): branch points from RSA task affinity of single-task nets, ImageNet-pretrained ResNet.
  - Learning to Branch (Guo, Lee, Ulbricht, ICML 2020, [PMLR](https://proceedings.mlr.press/v119/guo20e.html)).
  - AdaShare (Sun et al., NeurIPS 2020, [arXiv 1911.12423](https://arxiv.org/abs/1911.12423)).
  - TreeMTL (Zhang, Liu, Guan, AutoML-Conf 2022, [PMLR](https://proceedings.mlr.press/v188/zhang22a.html)).
- **Embedding Recycling** (Saad-Falcon et al., EACL Findings 2023, [arXiv 2207.04993](https://arxiv.org/abs/2207.04993)):
  - Caches layer-k activations and fine-tunes (or adapter-tunes) only the model's own layers k+1..N. The layers below k are never rerun.
  - Reports 87–91% inference speedup on DeBERTa-v2-XL with adapters.
  - The split layer is not optimised per task; the authors call the choice of what to recycle "a large decision space" left open.
- **Anthropic, "Cost-Effective Constitutional Classifiers via Representation Re-use"** (Cunningham et al., Alignment Science Blog, 2025, [link](https://alignment.anthropic.com/2025/cheap-monitors/)):
  - Retrains the final 1, 2, 4, … up to L/2 layers of the policy model, starting from its own weights. Earlier layers stay frozen and shared with the policy forward pass.
  - One retrained layer ranked second only to retraining the full model.
  - Linear probes cost about 0.1% overhead.

### Verdict: **already done**
The specific sub-question ("task-specific split depths with a FROZEN pretrained trunk where branches are initialised from copies of the base's own next layers") is answered by Wei et al. 2022:
- The teacher's task layers are the base's own top layers.
- Split depth varies per task.
- The trunk is the frozen pretrained BERT.
- One pass is shared across heterogeneous-depth branches.

Mainstream (2018) did the same for CNNs in a multi-tenant system.

**Remaining differences** (engineering or empirical, not architectural):
- Wei et al. branches always extend to layer 12 before KD. Tarski truncates at k+d with no teacher.
- ModernBERT instead of BERT.

### Queries tried
- "frozen pretrained transformer shared lower layers task-specific upper layers multi-task inference split layer"
- "frozen BERT shared bottom layers each task fine-tunes top layers serve many classifiers one forward pass branch point"
- "partial fine-tuning multi-task serving shared frozen layers BERT incremental tasks 2023 2024"
- "AdapterDrop … multiple tasks simultaneous inference share lower layers"
- "Mainstream dynamic stem-sharing multi-tenant … frozen layers fine-tune top layers"
- "stem sharing / shared stem transformer language model multi-tenant inference branch per application frozen layers"
- "shared trunk frozen encoder task branches independently trained added later without retraining text classification"
- Semantic Scholar citation graph of arXiv 2107.05377. The 9 citing papers include Embedding Recycling, "All Birds with One Stone", and "Deploying Multi-task Online Server with LLM" (COLING 2025 industry). None is a frozen-trunk heterogeneous-depth follow-up.
- GitHub: "CentraBert", "shared encoder", "multi-head classifier", "frozen backbone", "embedding recycling", "adapterdrop".

---

## Claim 2: Automatic per-task split depth; heterogeneous-depth branches from one trunk pass

### Closest prior work

- **Wei et al., ACL 2022.**
  - Per-task depth is chosen automatically in two stages:
    1. Validation-best L(τ) in [4, 10] (Sec. 3.1).
    2. "Ours (mixed)": keep adding task layers while the marginal gain is at least a threshold c (Sec. 4.2, Table 5; c = 1, 2, 3 trades accuracy for overhead).
  - The paper explicitly observes that tasks benefit from different depths: QQP reaches 91.0 with one task layer attached at frozen layer 2 (Sec. 5.1).
  - Its main stated limitation (Sec. 5.2) is about 10× more training for the search, typically 2–3 V100-days per task.
- **Mainstream, ATC 2018.**
  - M-Scheduler picks each app's branchpoint (degree of specialisation) and frame rate jointly, to maximise mean F1 under the edge compute budget.
  - It uses per-branchpoint accuracy profiles and per-layer cost measured once per hardware target (Sec. 6).
- **Probe-chosen layer, single task:**
  - LEC (Sawtell et al., Dec 2024, [arXiv 2412.13435](https://arxiv.org/abs/2412.13435)): logistic regression on the best intermediate layer, then prune the model there.
  - Skean et al., "Layer by Layer" (ICML 2025, [arXiv 2502.02013](https://arxiv.org/abs/2502.02013)): intermediate layers beat the last layer by 2–16% on MTEB.
  - Meyoyan & Del Corro, "A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification" (ACL 2026 main, [arXiv 2601.13288](https://arxiv.org/abs/2601.13288)): learned token and layer selection over the serving LLM's hidden-state tensor, several classifiers in the generation forward pass.
- **Automated branching in jointly trained MTL:** Vandenhende et al. 2020 (RSA affinities at candidate branch points), Learning to Branch 2020, AdaShare 2020 (per-task layer skip policy), TreeMTL 2022 (recommends a branch per task under a compute budget without training).
- **Cheap sharing-depth selection without training (vision edge):** Cao et al., "Representation Similarity: A Better Guidance of DNN Layer Sharing for Edge Computing without Training", MobiCom 2024 ([arXiv 2410.11233](https://arxiv.org/abs/2410.11233)).
- **Premise that tasks peak at different depths:**
  - Tenney, Das & Pavlick, "BERT Rediscovers the Classical NLP Pipeline", ACL 2019.
  - Liu et al., "Linguistic Knowledge and Transferability of Contextual Representations", NAACL 2019.
- **Early exit:** DeeBERT, PABEE, and F-PABEE ([arXiv 2305.11916](https://arxiv.org/abs/2305.11916)) choose the exit per *instance*, not per task. EE-Tuning ([arXiv 2402.00518](https://arxiv.org/abs/2402.00518)) adds exits to a frozen LLM. I found no NLP paper that fixes a static exit layer per *task* and serves many such tasks from one trunk, other than Wei et al.

### Gap
Nobody I found selects each task's fork depth from a cheap **layer-wise linear-probe sweep on cached frozen activations**, then serves the forked branches. Wei et al. use full partial-fine-tuning sweeps, Mainstream uses per-branchpoint fine-tuned models, and LEC picks a layer for a single task.

The obvious research question is untested in the literature: does the probe-optimal depth predict the fork-optimal depth? Per-request truncation of the trunk to the deepest *requested* branch is also not described. Wei et al. and Mainstream run every deployed task on every input, and Wei et al. list the resulting latency growth with N as a limitation (Sec. 5.2). This is a systems detail rather than a research claim.

### Verdict: **partially done**
Automatic per-task depth and heterogeneous-depth one-pass serving exist. The probe-guided selector, and evidence that it matches a fine-tuning sweep, would be new but narrow.

### Queries tried
- "per-task branch point selection frozen pretrained encoder layer-wise probe accuracy choose layer truncate multiple tasks"
- "task-specific exit layer multi-task transformer shared backbone static exit per task NLP classification"
- "multi-exit multi-task network task-specific exit layer transformer NLP"
- "multi-task early exit different tasks exit at different layers shared encoder task difficulty"
- "multi-task early exit task-level static exit layer chosen per task instead of per instance"
- "use linear probe accuracy per layer to decide how many layers to freeze fine-tuning depth selection proxy"
- "probing-guided layer selection truncation depth linear probe select layer then fine-tune top layers"
- "layer selection probe accuracy saturation cut transformer depth per task text classification ModernBERT"
- "attach each task head at its optimal intermediate layer frozen encoder multi-task heterogeneous depth single pass"
- AdaShare / Branched MTN / TreeMTL / AutoFreeze / Head2Toe follow-ups.

---

## Claim 3: Independently trained branches on a frozen backbone for many-tenant local serving

### Closest prior work
- **Wei et al., ACL 2022.**
  - Branches are trained independently, one task at a time.
  - The system is designed so updating, adding or removing a task leaves the others untouched.
  - In production, 21 tasks are owned by different teams (Appendix E).
  - Serving is a statically merged graph on datacenter GPUs, not dynamic loading.
- **Mainstream, ATC 2018.** Independently trained apps from different developers and multi-tenant edge execution with a shared stem. Deployment is scheduled centrally, with no per-request selection.
- **Tesla HydraNet** (Karpathy, CVPR 2021 workshop and Tesla AI Day 2021 talks; secondary summaries only, e.g. [thinkautonomous.ai](https://www.thinkautonomous.ai/blog/how-tesla-autopilot-works/)):
  - A shared backbone with many task heads.
  - Backbone features are cached and heads are fine-tuned on those cached features, independently of each other.
  - This is the canonical industrial precedent for "train heads on cached trunk activations". It is vision, with no split-depth choice, and not peer-reviewed.
- **Training on cached activations (NLP):** Embedding Recycling (EACL Findings 2023). AutoFreeze (Liu, Agarwal, Venkataraman, [arXiv 2102.01386](https://arxiv.org/abs/2102.01386)) freezes blocks adaptively and caches their outputs, giving up to 2.55× faster fine-tuning.
- **Frozen base with hot-swapped task modules:**
  - Apple Foundation Models (2024, [blog](https://machinelearning.apple.com/research/introducing-apple-foundation-models)): LoRA adapters are loaded dynamically, cached in memory and swapped on device; the base stays resident.
  - PetS (Zhou et al., USENIX ATC 2022).
  - S-LoRA (Sheng et al., MLSys 2024).
  - HMI (Zhang et al., [arXiv 2504.17449](https://arxiv.org/abs/2504.17449)): up to 10,000 tenant BERT/GPT variants on one GPU through parameter swapping.
  - Caveat: adapters in all layers share weights, not per-input computation.
- **Many probes on one forward pass (2025–26 guardrails and routing):**
  - Anthropic "cheap monitors" (2025) and Constitutional Classifiers++ ([arXiv 2601.04603](https://arxiv.org/abs/2601.04603), 2026).
  - Google DeepMind, "Building Production-Ready Probes For Gemini" ([arXiv 2601.11516](https://arxiv.org/abs/2601.11516), 2026), deployed.
  - McKenzie et al., "Detecting High-Stakes Interactions with Activation Probes" (NeurIPS 2025, [arXiv 2506.10805](https://arxiv.org/abs/2506.10805)): about six orders of magnitude cheaper than LLM monitors.
  - Meyoyan & Del Corro (ACL 2026).
  - LatentGate ([ACL 2026 Industry](https://aclanthology.org/2026.acl-industry.153/)): a frozen SLM plus linear probe router. About 28 ms on a T4, forward cost independent of the number of agents, and sub-10 ms warm-start probe retraining from feedback.
- **ModernBERT multi-decision router (industry):** vLLM Semantic Router.
  - The Oct 2025 blog ([vllm.ai](https://vllm.ai/blog/2025-10-27-semantic-router-modular)) presents intent, PII and jailbreak classifiers as one base pass plus LoRA adapters.
  - The project's own July 2026 issue ([#2394](https://github.com/vllm-project/semantic-router/issues/2394)) concedes that LoRA inside attention/MLP layers may still need separate per-task passes. It is an unmeasured proposal.
  - This is a real, current gap that tarski's split-depth design addresses directly.
- **Continual and modular learning:** Side-Tuning (Zhang et al., ECCV 2020), Progressive Nets, and PEFT-based task-incremental CL (e.g. MoCL, [arXiv 2404.00790](https://arxiv.org/abs/2404.00790)) all avoid interference with a frozen base. None shares per-input computation across tasks at different depths.
- **Industrial "one embedding, many heads":** Meta WPIE, a shared integrity representation feeding many classifiers ([Meta AI blog](https://ai.meta.com/blog/how-ai-is-getting-better-at-detecting-hate-speech/)). It uses only the final embedding.

### Gap
Not found: a local, consumer-hardware server where:
- end users train branches themselves on cached activations of a resident frozen encoder;
- branches of heterogeneous depth and type (probe or forked layers) are hot-loaded and evicted with an LRU;
- each request runs the trunk only to the deepest requested split.

Each piece exists separately:
- independent training and modularity: Wei, Mainstream;
- cached-feature head training: HydraNet, ER;
- dynamic module caching: Apple, S-LoRA;
- one-pass multi-probe: Anthropic, GDM, BERTology, LatentGate.

### Verdict: **partially done**
This is a systems-integration contribution, not a new idea. The vLLM Semantic Router issue is a useful motivating example: a ModernBERT multi-classifier stack where LoRA does not deliver true compute sharing.

### Queries tried
- "continual multi-task frozen backbone task-incremental heads"
- "task-incremental learning frozen pretrained language model independent task-specific modules no interference"
- "Side-Tuning … incremental learning frozen base"
- "PetS unified framework parameter-efficient transformers serving multi-task"
- "LoRA adapters only upper layers share lower layer computation across tasks multi-task inference same input prefix reuse"
- "Tesla HydraNet cache backbone features fine-tune task heads independently"
- "engineering blog shared text embedding many classifier heads one encoder pass production content moderation multi-head serving"
- "Meta Whole Post Integrity Embeddings WPIE shared representation many integrity classifiers"
- "arXiv 2025 one forward pass multiple probes classifiers shared hidden states guardrails"
- "Cost-Effective Constitutional Classifiers via Representation Re-use"
- "Google DeepMind production probes Gemini activation probes multiple classifiers 2026"
- "multiple linear probes different harm categories same residual stream activations single forward pass"
- "vLLM semantic router ModernBERT classifiers shared base multiple LoRA heads"
- "Apple foundation models adapters dynamically loaded cached swapped"
- "DroidSpeak KV cache sharing across fine-tuned LLMs" (NSDI 2026: cross-model reuse of lower-layer KV for fine-tuned variants; generative, not branches)
- "PrefillShare" (arXiv 2602.12029: frozen shared prefill module with task-specific decode modules, split by phase not by depth)
- GitHub: "frozen backbone" surfaced LatentGate.

---

## Claim 4: Latency/cost of N decisions per message: shared trunk vs N separate models vs question-in-input decision models, on consumer hardware

### Closest prior work
- **AdapterDrop, EMNLP 2021.**
  - Compares N fully fine-tuned models (sequential and fully parallel) against adapter models sharing lower layers, for N = 2 to 16 tasks.
  - Sequence length 128 and several batch sizes (Fig. 8, Table 2).
  - Hardware: GPU (V100 / Titan X).
- **Wei et al., ACL 2022.**
  - Layer-count overhead vs N separate models on GLUE.
  - Production P99 latency and GPU count for 21 tasks on V100 with FasterTransformer: 5 GPUs vs 42.
- **Latif, Zhou, Fang & Zhai, "Efficient Multi-Task Inference with a Shared Backbone and Lightweight Task-Specific Adapters for Automatic Scoring"** (AAAI iRAISE workshop, PMLR v273, 2025; [arXiv 2412.21065](https://arxiv.org/abs/2412.21065)).
  - 27 tasks with LoRA adapters.
  - GPU memory −60% and latency −40% vs separate fine-tuned models.
  - Quality: QWK 0.848 vs 0.888.
- **LatentGate (ACL 2026 Industry).** T4 latency vs prompt-based and embedding routers. Forward cost is independent of agent count.
- **Question-in-input decision models:**
  - Laya README ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)): per-question cost grows linearly on a T4 (see the sibling note `classification_multi_question.md`).
  - laya-mlx ([mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)): Apple M3 Max, about 13.4 ms median per short question (English) and about 7.4 ms (multilingual). The README explicitly says it does *not* encode the state once and reuse it across questions.
  - TypeSafe Jev: 70–500 ms end to end, architecture undisclosed ([docs](https://docs.typesafe.ai/introduction)).
- **Zero-shot label-in-input cost:** BTZSC (Aarab, ICLR 2026, [arXiv 2603.11991](https://arxiv.org/abs/2603.11991)) benchmarks accuracy and latency across NLI cross-encoders (one pass per label), embedding models, rerankers and LLMs.
- **Encode once, check many labels or policies:** GLiGuard and SCX Router (see sibling notes); CPU-class guard classifiers ([arXiv 2512.19011](https://arxiv.org/abs/2512.19011)).

### Gap
No study found that measures, on one message stream:
1. one partial trunk pass plus N branches;
2. N separately fine-tuned full encoders;
3. N question-in-input cross-encoder decisions (Laya/Jev style);

on Apple silicon or CPU, reporting per-decision marginal cost as N grows and branches have mixed depths. All shared-trunk measurements found are on datacenter GPUs. The only Apple-silicon decision-model numbers (laya-mlx) cover the per-question regime only.

### Verdict: **partially done**
The measurement type exists on datacenter GPUs. The specific three-way comparison on consumer hardware: **no prior work found**.

### Queries tried
- "latency N classifiers per message shared encoder vs separate fine-tuned models CPU Apple silicon benchmark multi-task BERT on-device"
- "on-device multi-task NLU shared encoder multiple heads latency memory smartphone single forward pass many tasks"
- "Efficient Multi-Task Inferencing Shared Backbone … latency GPU memory"
- "zero-shot NLI cross-encoder one forward pass per label cost vs shared encoding multiple labels"
- "BTZSC … throughput"
- "Laya decision model ModernBERT … GitHub", laya-mlx README
- "TypeSafe Jev System One typed decisions API"
- "Do You Really Need a GPU to Guard Your LLM"

---

## Claim 5: Initialise a branch from copies of the base's own next layers vs random-init head layers, frozen trunk below

### Closest prior work
- **Wei et al., ACL 2022.**
  - The teacher's task layers *are* the base's top layers, fine-tuned (partial fine-tuning).
  - The student, with x frozen and n task layers, is initialised from the bottommost N_s layers: layers [x, x+n) sit directly above the split, and everything above is dropped. That is truncate-and-fork, then trained by KD.
  - **Fig. 2 compares "vanilla initialization (from pre-trained BERT)" against teacher initialization for n ∈ {1, 2, 3}.** The paper adopts teacher initialization; the figure is an image and its magnitudes were not extractable.
  - There is no random-init arm.
- **MORES** (Gao, Dai, Callan, EMNLP 2020, [arXiv 2004.13313](https://arxiv.org/abs/2004.13313); details in the sibling note `decomposed_encoders_ir_qa.md`).
  - Interaction blocks are initialised from BERT's last l layers.
  - Randomly initialising the cross-attention instead of copying BERT self-attention drops MS MARCO MRR@10 from 0.3456 to 0.2723 (about −21% relative).
  - This is the strongest copy-vs-random evidence found, but the representation modules are fine-tuned, not a frozen raw pretrained trunk.
- **LUMEN** (de Jong et al., ICML 2023, [arXiv 2301.10448](https://arxiv.org/abs/2301.10448)): a frozen pretrained lower (1−α) "memory" encoder, with the "live" layers taken from the same model's top α layers and fine-tuned. α=1/3 is needed; α=1/8 recovers much less.
- **Anthropic cheap monitors (2025):** retraining the final k layers from the policy's own weights on a frozen trunk. The branch always runs to the model's end; no truncation.
- **EE-Tuning** (Pan et al., 2024, [arXiv 2402.00518](https://arxiv.org/abs/2402.00518)): early-exit layers added to a *frozen* LLM are initialised by copying modules of the original model, not randomly. This is the frozen-trunk, copy-init precedent in LLMs.
- **Mainstream (2018)** and **Embedding Recycling (2023):** specialised or upper layers are the base's own layers above the split. No truncation.
- **Freezing and dropping layers:**
  - Lee, Tang & Lin, "What Would Elsa Do? Freezing Layers During Transformer Fine-Tuning" ([arXiv 1911.03090](https://arxiv.org/abs/1911.03090), 2019): fine-tuning only the last quarter of layers gives about 90% of full quality.
  - Wei et al. Table 1: per-L partial fine-tuning curves on GLUE; 4–5 top layers reach about 99% on most tasks.
  - Sajjad, Dalvi, Durrani & Nakov, "On the Effect of Dropping Layers of Pre-trained Transformer Models" ("Poor Man's BERT", Computer Speech & Language 2023, [arXiv 2004.03844](https://arxiv.org/abs/2004.03844)): dropping top layers keeps up to 98% of performance at 40% pruning. Lower layers matter most.
- **Model growth by copying:** LLaMA Pro block expansion (Wu et al., ACL 2024, [arXiv 2401.02415](https://arxiv.org/abs/2401.02415)) copies blocks with zero-initialised output projections and trains only those, with the originals frozen. It adds depth rather than forking, but is the modern "copy layers, freeze the rest" reference.
- **Counter-evidence to cite:** Zhang et al., "Revisiting Few-sample BERT Fine-tuning" (ICLR 2021, [arXiv 2006.05987](https://arxiv.org/abs/2006.05987)) found that *re-initialising* BERT's top layers can improve few-sample fine-tuning. Pretrained top layers are not always the best initialisation, so the copy-vs-random ablation is not a foregone conclusion, especially with small end-user label sets.
- **Head initialisation:** Kumar et al., LP-FT (ICLR 2022, [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)) initialises the head from a trained linear probe before fine-tuning. Tarski's probe branch at depth k could seed the forked branch's head the same way.

### Gap
Not found: a controlled sweep over (k, d) on a frozen, raw-pretrained encoder that compares:
- (a) forked copies of layers [k, k+d) plus truncation;
- (b) d randomly initialised layers at k;
- (c) copies of the *final* d layers attached at k;
- (d) a probe at k (and at k+d).

MORES gives random vs copy only for cross-attention, in a non-frozen setting. Wei et al. compare pretrained vs teacher init, not random. Anthropic and EE-Tuning report copy-init only.

### Verdict: **already done** as an initialisation technique; **partially done** as an ablation
The systematic copy-vs-random-vs-probe comparison under a frozen trunk with truncation, at small-data end-user scale and on ModernBERT, would be a modest empirical contribution.

### Queries tried
- "EE-Tuning early-exit layers initialized copy from final layers frozen LLM"
- "classifier head transformer block initialized from pretrained next layer on frozen intermediate layer vs random initialized head comparison"
- "Revisiting Few-sample BERT Fine-tuning re-initializing top layers"
- "Poor Man's BERT dropping layers … top-layer dropping"
- "LLaMA Pro block expansion copy blocks zero-initialized output frozen original blocks"
- "What Would Elsa Do? Freezing Layers During Transformer Fine-Tuning"
- The full text of Wei et al. 2022, searched for "vanilla" and "initializ".
- MORES, LUMEN and DeFormer checked via the sibling notes.

---

## Other closely related works the thesis must cite (beyond the per-claim anchors)

1. **Saad-Falcon et al., "Embedding Recycling for Language Models"**, EACL Findings 2023. [arXiv 2207.04993](https://arxiv.org/abs/2207.04993) · [allenai/EmbeddingRecycling](https://github.com/allenai/EmbeddingRecycling). Caches a frozen intermediate layer and trains only the model's own upper layers on it. The closest precedent for "train branches on cached trunk activations", and it names split-layer choice as open.
2. **Meyoyan & Del Corro, "A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification"**, ACL 2026. [arXiv 2601.13288](https://arxiv.org/abs/2601.13288). Many classifiers in one serving pass with learned layer selection. The 2026 state of the art for the probe-branch half of tarski.
3. **Cunningham et al. (Anthropic), "Cost-Effective Constitutional Classifiers via Representation Re-use"**, 2025 ([link](https://alignment.anthropic.com/2025/cheap-monitors/)), plus **Constitutional Classifiers++** ([arXiv 2601.04603](https://arxiv.org/abs/2601.04603)). Probes vs retrained-final-layers vs standalone classifiers on shared activations, with cost accounting. The closest "probe vs fork" cost/accuracy comparison.
4. **LatentGate: Low-Latency Semantic Routing via Frozen-Backbone Probing of Small Language Models**, ACL 2026 Industry. [ACL Anthology](https://aclanthology.org/2026.acl-industry.153/). A frozen backbone plus probes for routing, with cost independent of the number of routes and fast warm-start retraining. The same application (routing) as tarski.
5. **Zhang et al., "Revisiting Few-sample BERT Fine-tuning"**, ICLR 2021. [arXiv 2006.05987](https://arxiv.org/abs/2006.05987). Counter-evidence for claim 5 that must be addressed.

Secondary works to cite:
- MT-DNN (ACL 2019).
- Tenney et al. (ACL 2019).
- Rethinking Hard-Parameter Sharing in Multi-Domain Learning (Zhang et al., ICME 2022, [arXiv 2107.11359](https://arxiv.org/abs/2107.11359)). Its caveat: for domain-shifted tenants, domain-specific *bottom* layers can beat shared bottoms.
- vLLM Semantic Router modular-LoRA blog and issue #2394.
- Latif et al. 2025.
- GLiGuard and SCX Router (sibling notes).

## Suggested positioning sentence

"Tarski revisits the frozen-shared-bottom, per-task-top-layers design of Wei et al. (ACL 2022) and Mainstream (ATC 2018) for local, user-trained decision serving. It contributes:
- a probe-guided split-depth selector that replaces a fine-tuning sweep;
- teacher-free truncate-and-fork branches trained on cached ModernBERT activations, with a copy-vs-random-vs-probe ablation;
- per-request trunk truncation with LRU branch hot-loading;
- a three-way cost comparison against question-in-input decision models (Laya/Jev) on Apple silicon and CPU."
