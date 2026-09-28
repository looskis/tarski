# Frozen LM Backbone + Appended Classifier Heads for Classification and Routing (Slack message routing, support-ticket categorization)

Scope note: covers "frozen LM/embedding model + small trained head" vs full fine-tuning, LoRA, and zero/few-shot LLM prompting for classification, intent detection and routing. Research date: 2026-09-27. Each finding is dated by its publication date. Other researchers cover adapters/side networks, storage/compute economics, and ReFT/steering, so those are touched only for context.

## 1. Linear probing / MLP heads on frozen features vs. full fine-tuning (incl. LP-FT)

### Takeaway
The in-distribution gap between a frozen backbone + linear head and full fine-tuning is usually small: about 2 points in Kumar et al. 2022, and roughly zero to a few F1 points in recent decoder-LLM studies. Frozen or lightly adapted features also generalize better out of distribution. The largest gains come from *some* supervised adaptation over zero-shot, not from full vs. partial adaptation. In 2025-26 the practical default for domain text classification is a classification head on LLM hidden states, adapted with LoRA if needed, not instruction-tuned generation.

### Cited Findings
- (Feb 2022) Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution": across their benchmarks, "fine-tuning obtains on average 2% higher accuracy ID but 7% lower accuracy OOD than linear probing". LP-FT (train the linear head first, then fine-tune everything) was "1% better ID, 10% better OOD than full fine-tuning". Mechanism: a random head causes gradients that distort the pretrained features. — [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)
- (Dec 2025, v3 May 2026) Yousefiramandi & Cooney, "Fine-Tuning Causal LLMs for Text Classification: Embedding-Based vs. Instruction-Based Approaches" — [arXiv 2512.12677](https://arxiv.org/html/2512.12677):
  - A *frozen* LLM2Vec embedding + head reached F1 = 0.833 on WIPO multi-label patent classification (14 classes). LoRA embedding-head reached 0.785, instruction tuning 0.819 and PatentBERT 0.801, so the frozen baseline was the strongest reported number on that task.
  - On single-label CLV (5 classes), the LoRA (r=8) embedding-head approach on Llama-3.2-3B reached F1 = 0.860 with 12.2M trainable params. Instruction-tuned Mistral-7B reached 0.853 with 167.8M, and fully fine-tuned PatentBERT 0.854 with 346M.
  - On AG News (4 classes), all working configurations clustered at F1 ≈ 0.92–0.93 with "no consistent winner".
  - Zero-shot Llama-3.2-3B-Instruct reached only F1 = 0.21 (CLV) and 0.17 (WIPO).
  - Throughput: inference at 1.88 samples/s for the Llama-3.2-3B embedding classifier, 0.23 for Mistral-7B instruction, and 93.45 for ModernBERT-base.
  - Distilling the 3B teacher into ModernBERT-base kept at least 97% of teacher F1 and was about 45× faster.
  - Calibration of the embedding-head classifier: ECE = 0.094. The authors say this is usable for confidence-based routing without temperature scaling.
  - Authors' recommendation: embedding-head LoRA adaptation is "a practical default for domain text classification under single-GPU constraints."
- (Aug 2024) Buckmann & Hill, "Logistic Regression makes small LLMs strong and explainable 'tens-of-shot' classifiers": penalized logistic regression on a small LLM's embeddings "equals (and usually betters) the performance of a large LLM" across 17 sentence-classification tasks (2–4 classes). It needs only about as many labels as it takes to validate a large LLM, and it gives stable explanations. — [arXiv 2408.03414](https://arxiv.org/abs/2408.03414)
- (Dec 2024) Sawtell et al., "Lightweight Safety Classification Using Pruned Language Models" (LEC): penalized logistic regression on an *intermediate-layer* hidden state of Qwen 2.5 0.5B/1.5B/3B (and DeBERTa v3). With fewer than 100 examples it surpassed GPT-4o and special-purpose fine-tuned models on content-safety and prompt-injection classification. — [arXiv 2412.13435](https://arxiv.org/abs/2412.13435)
- (Nov 2025) Counter-example (IT tickets, EasyVista): on about 26k real multilingual ITSM tickets for priority prediction, a pipeline of frozen multilingual-mpnet embeddings + clustering + XGBoost/RF reached only about 24% combined test F1. A fine-tuned XLM-RoBERTa with numeric features reached 78.5% average F1. Caveat: the "embedding" arm was mostly unsupervised clustering with label assignment, not a proper supervised head, so it is a weak baseline, not a clean LP-vs-FT comparison. — [arXiv 2512.17916](https://arxiv.org/html/2512.17916v1)

### Inferences
- For routing and ticket categories with clean labels and at least tens of examples per class, a frozen-feature head should land within a few points of full fine-tuning. If the gap matters, LP-FT or LoRA-on-backbone plus head closes it at small parameter cost.
- A head trained on frozen features is likely *more* robust to distribution shift (new teams, new phrasing, seasonal drift) than a fully fine-tuned model, per Kumar et al. This matters for Slack routing, where message style drifts.
- The weak results in poorly designed "embedding" pipelines (2512.17916) suggest the head must be supervised: logistic regression or an MLP on labels, not clustering.

### Gaps
- I did not find a recent, clean head-to-head of frozen decoder-LLM linear probes vs. full fine-tuning on the standard intent sets (CLINC150/BANKING77/HWU64) with 2025-26 models. Most intent numbers compare fine-tuned encoders against prompted LLMs.
- I did not find GLUE-style LP-vs-FT gap tables for modern decoder LLMs.

## 2. Which layer to tap, and pooling

### Takeaway
For decoder LLMs, mid-depth layers usually give better classification features than the final layer, with average gains of 2–16% on MTEB tasks. Pruning the model at the best intermediate layer also cuts compute. For adapted or trained embedding models, last-token and mean pooling perform about the same. For raw frozen decoder LLMs, the Layer-by-Layer analysis used mean pooling.

### Cited Findings
- (Feb 2025, ICML 2025) Skean et al., "Layer by Layer: Uncovering Hidden Representations in Language Models": across 32 MTEB tasks (classification, clustering, reranking) and models including Pythia, Llama3, Mamba, BERT and LLM2Vec, intermediate layers beat the final layer by "2% to as high as 16% on average". The best layer "often resides around the mid-depth." Autoregressive models show a pronounced mid-layer "compression valley" in entropy, while bidirectional models (BERT) change less across layers. The analysis used mean token pooling. — [arXiv 2502.02013](https://arxiv.org/html/2502.02013); [ICML 2025 proceedings](https://proceedings.mlr.press/v267/skean25a.html)
- (Dec 2024, NeurIPS 2024 workshop) Skean, Arefin, LeCun, Shwartz-Ziv, "Does Representation Matter? Exploring Intermediate Layers in LLMs" (precursor): intermediate layers "often yield more informative representations for downstream tasks than the final layers". Transformers and SSMs differ architecturally, and some intermediate layers show bimodal entropy. — [arXiv 2412.09563](https://arxiv.org/abs/2412.09563)
- (Dec 2024) LEC: intermediate transformer layers "consistently outperformed final layers" for both safety tasks. The model can be "pruned to the optimal intermediate layer and used exclusively as robust feature extractors." — [arXiv 2412.13435](https://arxiv.org/abs/2412.13435)
- (2025-26) With LoRA-adapted decoder LLMs, last-token pooling matched mean pooling within ≤0.014 F1. Max pooling and learnable [CLS] tokens showed failures, especially on multi-label tasks. — [arXiv 2512.12677](https://arxiv.org/html/2512.12677)

### Inferences
- Practical recipe: extract hidden states at several depths (for example 25/50/75/100%) and cross-validate a logistic-regression head per layer. Then truncate the model at the winning layer. For a 32-layer model this can roughly halve per-message compute, since the layers above are never run.
- Mean pooling is the safer default for a raw frozen decoder LLM, because last-token features of an unadapted causal model depend heavily on the prompt. Last-token pooling works once the model is adapted or is an embedding model trained with last-token (EOS) pooling, as the Qwen3/E5-Mistral-style models are.

### Gaps
- There is no consensus rule for the exact optimal depth per model family. It varies by architecture and task and has to be picked empirically.
- I found no study of layer choice specifically on short, informal chat messages such as Slack.

## 3. Decoder LLMs as frozen embedding/feature extractors (LLM2Vec, E5-Mistral, NV-Embed, GritLM, Qwen3-Embedding)

### Takeaway
Off-the-shelf LLM-based embedding models, frozen with a logistic-regression head, are strong classifiers. MTEB's classification category measures exactly this protocol. As of mid-2025, Qwen3-Embedding-8B and Gemini embedding lead: about 90 average accuracy on MTEB English v2 classification and 72–74 on MTEB multilingual. The 0.6B Qwen3 model gets within about 5 points of the 8B.

### Cited Findings
- MTEB's Classification tasks embed the train and test sets with the frozen model and train a logistic-regression classifier on the train embeddings, so these scores are a direct "frozen embedder + linear head" benchmark. — [MTEB paper, arXiv 2210.07316](https://arxiv.org/abs/2210.07316)
- (June 2025) Qwen3-Embedding MTEB classification scores from the official repo — [QwenLM/Qwen3-Embedding GitHub](https://github.com/QwenLM/Qwen3-Embedding):
  - MTEB English v2, Classification (Overall mean in parentheses):

    | Model | Classification (overall mean) |
    |---|---|
    | Qwen3-Embedding-8B | 90.43 (75.22) |
    | Qwen3-Embedding-4B | 89.84 (74.60) |
    | Qwen3-Embedding-0.6B | 85.76 (70.70) |
    | gemini-embedding-exp-03-07 | 90.05 (73.3) |
    | gte-Qwen2-7B-instruct | 88.52 |
    | NV-Embed-v2 (7.8B) | 87.19 (69.81) |
    | multilingual-e5-large-instruct (0.6B) | 75.54 |

  - MTEB Multilingual, Classification:

    | Model | Classification |
    |---|---|
    | Qwen3-8B | 74.00 |
    | Qwen3-4B | 72.33 |
    | Qwen3-0.6B | 66.83 |
    | gemini-embedding-exp | 71.82 |
    | multilingual-e5-large-instruct | 64.94 |
    | gte-Qwen2-7B | 61.55 |
    | text-embedding-3-large | 60.27 |
    | NV-Embed-v2 | 57.29 |

  - Qwen3-Embedding-8B ranked #1 on MTEB multilingual (70.58) as of June 5, 2025.
  - Note: a secondary blog lists different Qwen3 "classification" numbers (76.97 / 75.46 / 71.40), probably from a different MTEB variant. Prefer the primary GitHub table. — [premai.io blog](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/)
- (Apr 2024) LLM2Vec converts a causal LLM into an encoder in three steps (bidirectional attention, masked next-token prediction, unsupervised contrastive learning). It reached state-of-the-art MTEB among public-data-only models as of May 2024. A search-result snippet gives MTEB classification averages of about 76.3 (LLaMA-2-7B) and 76.6 (Mistral-7B) for LLM2Vec variants; which variant (unsupervised vs. supervised) was not verified. — [arXiv 2404.05961](https://arxiv.org/abs/2404.05961)
- (Dec 2025/May 2026) A frozen LLM2Vec encoder + head beat LoRA and instruction-tuned variants on a 14-class multi-label task (F1 0.833), which is direct evidence that frozen LLM embeddings are competitive for domain classification. — [arXiv 2512.12677](https://arxiv.org/html/2512.12677)

### Inferences
- For a Slack or ticket router, a frozen 0.6B–8B embedding model plus logistic regression is the minimal "appended layer" approach, and its benchmark accuracy is already near fine-tuned levels on MTEB-style tasks. Model choice matters by about 5 points (0.6B vs 8B) and matters more for multilingual traffic. NV-Embed-v2 is English-focused and drops to 57 on multilingual classification.
- Instruction-aware embedders (Qwen3, E5-Mistral, GritLM) accept a task instruction prefix, such as "Classify the team responsible for this message". Changing that instruction can re-target one frozen model for different classification tasks without retraining.

### Gaps
- I did not retrieve 2026 MTEB leaderboard snapshots, so newer models such as Qwen3-VL-Embedding, KaLM-V2 or Gemini updates are not covered, and the current #1 for classification may have changed.
- I did not verify NV-Embed/GritLM/E5-Mistral classification scores from their own papers (the only numbers above come from the Qwen3 table).

## 4. Few-shot efficient approaches: SetFit, prototypes/centroids, kNN, semantic-router style routing

### Takeaway
SetFit (contrastively fine-tune a small sentence transformer on pairs, then fit a logistic-regression head) matched RoBERTa-large fine-tuned on 3k examples using only 8 examples per class, and trains in about 30 s for about $0.03. Centroid/prototype and kNN classifiers over frozen embeddings are nearly as accurate as SVMs on ticket data. Their key operational advantage is adding or changing classes without retraining.

### Cited Findings
- (Sep 2022) Hugging Face SetFit blog — [HF blog](https://huggingface.co/blog/setfit):
  - With 8 labeled examples per class on Customer Reviews, SetFit was competitive with RoBERTa-large fine-tuned on 3k examples.
  - RAFT leaderboard: SetFit (RoBERTa-large, 355M) 71.3%, T-Few (11B) 75.8%, GPT-3 (175B) 62.7%, human baseline 73.5%. SetFit beat humans on 7 of 11 tasks.
  - Training cost: 30 s on a V100 for about $0.025, vs. 11 min for about $0.7 for T-Few 3B on an A100 (28× faster and cheaper). The MPNet backbone variant has 110M params.
  - Works multilingually.
- (2021) CPFT few-shot intent detection (contrastive pre-training + fine-tuning): BANKING77 accuracy 80.86% at 5-shot and 87.20% at 10-shot; HWU64 82.03% at 5-shot and 87.13% at 10-shot. — [ACL Anthology EMNLP 2021](https://aclanthology.org/2021.emnlp-main.144.pdf)
- (2023) QAID improves over CPFT by more than 4 points on BANKING77 5-shot, with 8.5% / 18.9% error-rate reduction on CLINC150 / HWU64. — [arXiv 2303.01593](https://arxiv.org/pdf/2303.01593)
- (Nov 2025) Mohanna & Ait-Bachir (EasyVista), centroid-based ITSM classification — [arXiv 2511.20667](https://arxiv.org/abs/2511.20667):
  - Data: 8,968 real ITSM tickets, 123 categories.
  - Method: separate semantic-embedding and lexical centroids per category, fused by reciprocal rank fusion.
  - Accuracy: hierarchical F1 0.731 vs. 0.727 for an SVM baseline.
  - Speed: training 5.9× faster than SVM; incremental updates up to 152× faster than retraining; batch inference 8.6–8.8× faster.
  - Centroids also give interpretability.
- (June 2026) Sentence-embedding kNN (all-MiniLM-L6-v2, cosine) was a baseline in the production intent-detection comparison (Section 5). The fetched summary reported headline numbers mainly for RoBERTa vs. Claude. — [arXiv 2608.20371](https://arxiv.org/html/2608.20371)
- Practitioner caution (low-authority dev.to posts, 2025-26): embedding similarity can fail via "semantic shortcuts". For example, "refund pending" vs. "refund requested" share vocabulary but have different owners and SLAs. The suggested pattern is an embeddings classifier for routine labels, escalation of uncertain cases to an LLM, and reranking for close candidates. — [dev.to edtech ticket tagging](https://dev.to/xaviorcross6845/edtech-support-ticket-tagging-a-fine-tuning-alternative-with-auditable-tenant-cost-30j9); [dev.to fintech triage](https://dev.to/jensencole5829/fintech-support-triage-embeddings-classifiers-or-zero-shot-llms-20g5)

### Inferences
- For a Slack router with tens of destinations and a handful of examples each, the best starting point is SetFit, or a frozen embedder + logistic regression. A centroid/kNN index is the right choice where destinations change often.
- Semantic-router-style libraries (an utterance list per route, cosine similarity with a threshold) are essentially the prototype/kNN approach. Evidence on ticket data (0.731 vs 0.727 hierarchical F1) suggests they sacrifice little accuracy vs. trained linear heads.

### Gaps
- I did not find an independent benchmark of the "semantic-router" library itself, or direct SetFit-vs-frozen-LLM-embedding-head comparisons on 2025-26 embedders.
- I did not find published label-efficiency curves showing how many labels per class a frozen Qwen3/E5 embedder + LR needs to match fine-tuning on intent benchmarks.

## 5. Frozen-backbone classifiers vs. zero/few-shot LLM prompting: accuracy, cost, latency; industry case studies

### Takeaway
With abundant labels and a stable, narrow label schema, fine-tuned or frozen-feature classifiers beat zero-shot LLMs by about 10–25 points. On ATIS the gap was 95.9% vs. 84.1%. On broad schemas (CLINC150) the two tie (89.1% vs. 88.5%), and LLMs are much better at out-of-scope detection and at schemas that change per deployment. LLM calls are about 400–1,000× slower (about 1 s vs. 2.4 ms). Open ≤9B models used zero-shot do poorly on fine-grained intent sets (39% on BANKING77). The recommended production pattern is a hybrid: a cheap classifier first, with an LLM fallback for low-confidence or out-of-scope cases.

### Cited Findings
- (June 2026) Rodrigues & Vas, "When Do LLMs Replace Fine-Tuned NLU? A Decision Framework for Intent Detection in Production Conversational Systems" — [arXiv 2608.20371](https://arxiv.org/html/2608.20371):
  - In-scope accuracy:

    | Dataset | Fine-tuned RoBERTa-base | Claude Haiku, zero-shot | Significance |
    |---|---|---|---|
    | ATIS (26 intents) | 95.9% | 84.1% | p < 0.001 |
    | CLINC150 (150 intents) | 89.1% | 88.5% | p = 0.24, not significant |

  - CLINC150 out-of-scope detection:

    | Model | OOS recall | OOS F1 |
    |---|---|---|
    | Claude | 85.6% | 85.0% |
    | RoBERTa | 58.1% | 73.0% |
    | TF-IDF | 36.4% | 51.7% |

  - Dynamic schema: a RoBERTa locked to App A scored 96.8% on App A but **0% on an unseen App B**. Schema-prompted Claude scored 94.3% on A and 93.9% on B.
  - Noisy ASR input (28.9% WER): Claude 92.5% vs. TF-IDF 80.0%.
  - Latency and cost:

    | Model | Latency | Cost per 1K queries |
    |---|---|---|
    | RoBERTa | 2.4 ms p50 | $0 |
    | Claude | 981 ms p50, 1,787 ms p95 | $0.25 |

  - Decision framework: stable schema + abundant labels + latency-critical → fine-tuned encoder. Dynamic or per-deployment schemas with few labels → schema-prompted LLM. OOS-critical or noisy input → LLM or hybrid (fast classifier + LLM fallback).
- (July 2026) Ganesh, Dozier, Seals (Auburn), 41 open-weight LLMs (135M–9B) for zero-shot intent classification — [arXiv 2607.27421](https://arxiv.org/html/2607.27421):
  - Average accuracy across models: CLINC150 0.468, BANKING77 0.388, MASSIVE 0.432, MTOP 0.348, SNIPS 0.807 (saturated).
  - Best aggregate: Mistral-7B-Instruct-v0.3 at 0.660.
  - Instruction tuning mattered more than scale; a 3B instruct model beat several 7B base models.
  - Qwen2.5-3B-Instruct was the best sub-4B model: 0.632 at 20.8 ms/sample and 5.8 GB VRAM.
  - Typos were the most damaging perturbation.
- (Dec 2025/May 2026) Zero-shot Llama-3.2-3B-Instruct scored F1 0.21 / 0.17 on domain classification vs. 0.86 / 0.79 after embedding-head LoRA. — [arXiv 2512.12677](https://arxiv.org/html/2512.12677)
- (Unverified hobby repo, via search snippet) On BANKING77, zero-shot Qwen2.5-1.5B scored 36.9% vs. 93.4% after QLoRA. Prompted models were described as about 35–70× slower per query than batched local adapters, at about $2 per 1K queries. — [GitHub finetune-intent-lab](https://github.com/RahulRachhoya/finetune-intent-lab)
- (June 2024, rev. Feb 2025) RouteLLM (LMSYS/Berkeley/Anyscale) — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665):
  - The task: route queries between a strong and a weak LLM.
  - Router designs: a BERT classifier, matrix factorization, a causal-LLM classifier, and similarity-weighted ranking.
  - Results: cost cut "by over 2 times in certain cases" without quality loss, and routers generalized when the model pair changed at test time.
  - This is an instance of "small classifier on text features decides the destination."
- (Nov 2025) Real ITSM ticket categorization (123 classes): an embedding-centroid classifier matched an SVM (Section 4). No LLM arm was reported in the abstract. — [arXiv 2511.20667](https://arxiv.org/abs/2511.20667)
- Hybrid OOS pattern (2025): an uncertainty score on an in-scope classifier triggers a (fine-tuned) LLM to correct the prediction when an utterance looks out-of-scope or ambiguous. — [arXiv 2507.22289](https://arxiv.org/pdf/2507.22289) (search-snippet level only)

### Inferences
- A Slack router is usually a "broad, evolving schema with moderate labels" problem, which is where zero-shot LLMs and trained classifiers tie on accuracy. A frozen-embedding head then wins on cost and latency by about three orders of magnitude, and an LLM fallback covers OOS and newly added destinations.
- Small open models used zero-shot (≤9B) are a poor substitute for either a trained head or a frontier API on fine-grained label sets (39–47% on BANKING77/CLINC150). The same small models used as *frozen feature extractors with a head* are far stronger.

### Gaps
- No published engineering case studies from Zendesk, Intercom, Atlassian, Uber, Airbnb or Dropbox were retrieved in this pass. I therefore cannot cite company-reported accuracy numbers for production ticket or message routing.
- I found no peer-reviewed work on Slack-message routing specifically. Only hobby or tutorial material exists, such as a Chariot Solutions blog on a neural net that predicts the Slack channel for a message ([Chariot Solutions](https://chariotsolutions.com/blog/post/neural-network-to-classify-slack-messages/)) and Zapier/LangGraph triage templates, none with metrics.
- A search snippet claimed that fine-tuned PLMs did not significantly beat linear SVMs on real IT tickets and that LLMs were "590× slower". I could not attribute this to a specific source, so it is excluded.

## 6. Multi-task / multi-tenant: one frozen backbone, many heads; adding classes; out-of-scope detection

### Takeaway
A frozen backbone decouples representation from label space. Many heads (per task or per tenant) can share one embedding pass, and new classes can be added by adding a centroid or retraining a logistic-regression head in seconds, instead of re-fine-tuning. Fully fine-tuned, closed-set classifiers break on schema change: 0% on unseen intents in one study. They also detect out-of-scope inputs worse than LLMs (OOS F1 73% vs. 85%).

### Cited Findings
- (Dec 2024) A single general-purpose small LM served two classification tasks (content safety and prompt injection) with separate LR heads on intermediate hidden states. — [arXiv 2412.13435](https://arxiv.org/abs/2412.13435)
- (Nov 2025) Centroid classifier: incremental updates up to 152× faster than retraining, which suits new categories. — [arXiv 2511.20667](https://arxiv.org/abs/2511.20667)
- (June 2026) A locked fine-tuned RoBERTa scored 0% on an unseen app schema, while a schema-prompted LLM stayed at 93.9%. RoBERTa OOS F1 was 73.0% vs. 85.0% for Claude on CLINC150. — [arXiv 2608.20371](https://arxiv.org/html/2608.20371)
- (July 2026) OOS detection on frozen MiniLM embeddings via multi-cluster boundary learning, evaluated on CLINC150, StackOverflow and BANKING77. This is an embedding-space open-set method that needs no backbone retraining. — [arXiv 2607.07974](https://arxiv.org/html/2607.07974) (details not fetched)
- (2025) DROID "Dual Representation for Out-of-Scope Intent Detection" — [arXiv 2510.14110](https://arxiv.org/pdf/2510.14110) — and an ACL 2025 Industry paper on efficient OOS detection — [ACL Anthology](https://aclanthology.org/2025.acl-industry.25.pdf) — exist, but their numbers were not retrieved.
- (2022) Continual few-shot intent detection is an established research line (COLING 2022). — [ACL Anthology](https://aclanthology.org/2022.coling-1.26.pdf)
- (2023) Multi-tenant few-shot FAQ retrieval shows a multi-tenant embedding approach. — [arXiv 2301.10517](https://arxiv.org/pdf/2301.10517) (not fetched)

### Inferences
- Architecture for multi-team or multi-workspace routing:
  1. Run one frozen embedder, or one truncated LLM at its best layer, once per message.
  2. Cache the vector.
  3. Attach N cheap heads: team router, ticket category, urgency, and OOS/"none of the above".
  4. Add a new destination by adding examples → centroid/LR refit in seconds. There is no GPU retraining and backbone features stay stable.
- OOS handling should be explicit: a distance or energy threshold, or a "none" class trained with negative examples. Escalate to an LLM when the head is uncertain, since closed-set heads are weakest there.

### Gaps
- I did not find quantitative studies on how many simultaneous heads or tenants one frozen backbone can serve before per-tenant accuracy drops. This is mainly a question for the storage/compute researcher.
- OOS numbers for frozen-LLM-embedding heads (vs. fine-tuned encoders) on CLINC150 were not retrieved.

## 7. Early-exit / intermediate-layer classifiers that skip the rest of the model

### Takeaway
For *classification*, the simplest form of early exit is static: cut the model at the best intermediate layer and put a linear head there. The evidence supports this (LEC, Layer-by-Layer). Dynamic per-token early exit is becoming less effective in newer, less redundant LLMs.

### Cited Findings
- (Dec 2024) LEC: prune a small LM to its optimal intermediate layer and use it only as a feature extractor + LR head. This beat GPT-4o on safety and injection classification with fewer than 100 examples. — [arXiv 2412.13435](https://arxiv.org/abs/2412.13435)
- (Feb 2025) Mid-depth layers outperform final layers on MTEB tasks by 2–16% on average, so truncating at mid-depth can improve accuracy *and* roughly halve the forward cost. — [arXiv 2502.02013](https://arxiv.org/html/2502.02013)
- (Apr 2024) LayerSkip (Meta): training with layer dropout and an early-exit loss enables early-exit inference and self-speculative decoding without auxiliary heads. The abstract reports speedups up to about 2× on generation tasks, including TOPv2 semantic parsing. — [arXiv 2404.16710](https://arxiv.org/abs/2404.16710)
- (Mar 2026) Wei et al., "The Diminishing Returns of Early-Exit Decoding in Modern LLMs": early-exit effectiveness declines across newer model generations because improved pretraining recipes and architectures "reduce layer redundancy". Dense transformers show more early-exit potential than MoE or SSMs, >20B models more than smaller ones, and base models more than specialized fine-tunes. — [arXiv 2603.23701](https://arxiv.org/abs/2603.23701)
- (2025) Classification-adapter early exit, one lightweight classifier per layer (SVM reported as a good choice), is an active line. Sentence-by-sentence incremental processing gave up to 2.3× speedup on sentiment classification. — search-result summary; primary source not verified ([arXiv 2604.18592](https://arxiv.org/html/2604.18592v1))

### Inferences
- For routing short Slack messages, static truncation at the best probed layer is simpler and more predictable than dynamic confidence-based exits. Short inputs leave little per-token savings, and newer models are less redundant.
- Dynamic early exit among the *heads* still makes sense in a cascade: a cheap truncated-backbone head answers confident cases, and uncertain ones go to the full model or an LLM. This is the same hybrid pattern as Section 5.

### Gaps
- I found no benchmark of early-exit classifiers specifically on intent or ticket datasets with 2025-26 LLMs.
- The 2.3× incremental-processing claim and the "SVM is optimal" claim come from search summaries and were not verified against the primary text.
