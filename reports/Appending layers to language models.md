# Freeze the Backbone, Append the Task

The answer to both questions is yes. Researchers have been adding trainable layers to a frozen language model, instead of changing its weights, since 2019. Bottleneck adapters inserted into a frozen BERT came **within 0.4 points of full fine-tuning on GLUE while adding 3.6% new parameters per task**. Eight new blocks interleaved into a frozen LLaMA2-7B **more than doubled its HumanEval score** and left general benchmarks roughly unchanged. For single-message classification, a small head trained on the frozen model's middle-layer features scores within a few points of full fine-tuning and holds up better when the input distribution shifts.

The savings are large, and they grow with the number of tasks. A full fine-tune of a 7B model is about **14.5 GB per task**, a LoRA adapter about **14 MB**, and a linear classification head about **80 KB**. So 1,000 per-team routers on one shared backbone take up about as much storage as two model copies. One cached forward pass can also feed many heads, each of which trains in seconds.

The savings disappear in three places:
- **Latency.** Added depth that cannot be merged into existing weights costs **20–30% extra latency at batch size 1**.
- **Top-only limits.** A head appended only at the top cannot recover information the backbone has already discarded.
- **Upgrades.** Every appended module is tied to its backbone. Naively copying an intent adapter to the next Qwen release **dropped Banking77 accuracy from 92.8% to 42.9%**.

For Slack routing and ticket triage, the practical design is one frozen embedder, or an LLM truncated at a middle layer, with many cheap heads on top and an LLM fallback for messages that fit no route. Appending layers is not new in itself. The genuinely novel openings are:
- routers that survive backbone upgrades;
- hypernetworks that generate a classifier from a label taxonomy;
- measured serving of many classifiers from one forward pass;
- classifiers trained with reinforcement learning that use only tens of parameters.

None of the literature surveyed here addresses these for classification.

## Appending layers has worked since 2019, and placement decides quality

The PEFT (parameter-efficient fine-tuning) literature already has a name for this idea. **"Additive" methods "introduce new parameters or layers while freezing original weights."** Surveys set them apart from selective methods, which tune a subset of existing weights, and from reparameterization methods such as LoRA ([Lialin et al. 2023](https://ar5iv.labs.arxiv.org/html/2303.15647)).

The founding result is the Houlsby adapter: a small bottleneck module, inserted twice into every layer of a frozen BERT. Houlsby adapters reached **80.0 on GLUE against 80.4 for full fine-tuning**. Across nine tasks they needed **1.3× the base model's parameters in total, against 9× for nine fine-tuned copies** ([Houlsby et al. 2019](https://ar5iv.labs.arxiv.org/html/1902.00751)). Later work cut the budget and widened the scope. Compacter matched full fine-tuning on GLUE while training **0.047% of parameters** ([Mahabadi et al. 2021](https://arxiv.org/abs/2106.04647)). On a decoder LLM, LLaMA Pro trained only eight new identity-initialized blocks added to a frozen LLaMA2-7B. That moved **HumanEval from 13.05 to 28.66 and MBPP from 20.09 to 33.20**, while ARC, HellaSwag, MMLU and Winogrande each moved by about a point ([Wu et al. 2024](https://arxiv.org/html/2401.02415)).

| Method (year) | Where new parameters sit | New params | Result vs. full fine-tuning (FT) |
|---|---|---|---|
| Houlsby adapters (2019) | Two bottlenecks inside every layer | 3.6%/task | GLUE 80.0 vs 80.4 ([source](https://arxiv.org/abs/1902.00751)) |
| Parallel/MAM adapter (2021) | Parallel to the FFN, plus a prefix | 0.5% (GLUE), 6.7% (XSum) | MNLI/SST-2 match FT; XSum R-2 21.90 vs 21.94 ([source](https://ar5iv.labs.arxiv.org/html/2110.04366)) |
| LLaMA Pro (2024) | 8 new blocks interleaved into 32 frozen blocks | 7B → 8.3B | Code/math gains, general benchmarks flat ([source](https://arxiv.org/html/2401.02415)) |
| Ladder Side-Tuning (2022) | Thin side network reading every layer | 1.74% | GLUE 84.1 vs 85.2, at 5.5 GB vs 17.6 GB memory ([source](https://ar5iv.labs.arxiv.org/html/2206.06522)) |
| Quantized Side Tuning (2024) | Side network on a 4-bit 70B model | small | MMLU 63.9 vs QLoRA's 64.1, at 56 GB vs 95.5 GB ([source](https://arxiv.org/html/2401.07159)) |
| LoReFT (2024) | Low-rank edits to hidden states | 0.015% | GLUE (RoBERTa-large) 88.2 vs LoRA 88.1 ([source](https://arxiv.org/html/2404.03592v3)) |
| Proxy-tuning (2024) | A whole small tuned model steering the big model's output logits | 7B helper | Closes 88% of the 70B base-to-chat gap ([source](https://arxiv.org/html/2401.08565)) |

The "add versus adjust" framing hides the distinction that matters in practice. LoRA also leaves the original weights untouched: it learns a separate low-rank delta. The real divide is whether the new parameters can be *merged* into existing matrices.
- **Merged LoRA** adds no inference latency, but the LoRA authors note that once adapters are absorbed into the weights, inputs for different tasks are no longer straightforward to batch in one forward pass ([Hu et al. 2021](https://ar5iv.labs.arxiv.org/html/2106.09685)).
- **Truly appended modules** (adapters, blocks, side networks, heads) cannot be folded away, so they cost extra compute. In exchange, they leave the base model bit-for-bit intact, can be removed to recover it exactly, and can share one forward pass with other tasks' modules.

For routing and triage, where many small tasks read the same text, that composability is what you are paying for.

**Where the new parameters go matters as much as how many there are.**
- **Wiring:** He et al. recast adapters, prefix tuning and LoRA as edits to hidden states. They found that "parallel adapter is able to beat sequential adapters in all cases" and that edits beside the FFN beat edits beside attention ([He et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.04366)).
- **Depth:** LLaMA Pro's placement test is the cleanest single data point for the "append" question. On the general-task average, eight new blocks scored **56.70 interleaved, 56.19 stacked on top, and 29.36 stacked at the bottom**. Bottom placement broke the model ([Wu et al. 2024](https://arxiv.org/html/2401.02415)). The explanation is that anything inserted low feeds shifted inputs to every frozen layer above it.
- **Initialization:** zero-initializing each new block's output projections is what makes insertion safe, because training then starts exactly at the base model's behavior.
- **Why middle insertion is safe:** the middle layers of pretrained transformers show "a surprising amount of uniformity," and models tolerate middle layers being skipped, reordered or duplicated ([Sun et al. 2024](https://arxiv.org/abs/2407.09298)).
- **Low layers matter least:** removing adapters from the first five layers keeps **95–97% of GLUE performance** ([Rücklé et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.11918)).

A head appended only at the very top is the weakest form of the idea. Freezing everything in BERT or RoBERTa except the output layer recovers only about **64% of full-model quality on GLUE** ([Lee et al. 2019](https://ar5iv.labs.arxiv.org/html/1911.03090)). Matching adapter quality by fine-tuning only the top layers took **52.9% of parameters per task, against 1.14% for adapters spread through depth** ([Houlsby et al. 2019](https://ar5iv.labs.arxiv.org/html/1902.00751)). Theory explains why.
- A module that sees only the final hidden state can only re-read what that layer kept.
- Prompt and prefix methods "cannot change the relative attention pattern over the content." They only bias attention outputs in a fixed direction, so they bring out existing skills rather than teach new ones ([Petrov et al. 2023](https://arxiv.org/abs/2310.19698)).

The evidence supports a rough ranking by expressive power, from least to most:
1. a head on final features;
2. prompts or prefixes;
3. side networks that read every layer;
4. adapters or blocks inserted into the forward path;
5. direct weight changes.

This ranking is my synthesis rather than a proven theorem. It matches the accuracy ordering on generation tasks, where prefix < sequential adapter < parallel adapter ≈ full fine-tuning ([He et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.04366)).

The ranking also explains where additive methods fall short. On summarization and translation, closing the gap to full fine-tuning takes a **6.7% parameter budget**, against 0.5% for GLUE-style classification ([He et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.04366)). LoReFT matches LoRA on classification but trails it on arithmetic reasoning, which the authors attribute to long chain-of-thought generation ([Wu et al. 2024](https://arxiv.org/html/2404.03592v3)). The newest additive block methods therefore focus on *which* layers to expand. ADEPT duplicates the layers least important to general competence and reports up to **+5.58% on target domains while tuning 15% of parameters**, compared with full continued pretraining ([ADEPT 2025](https://arxiv.org/abs/2510.10071)).

## Routing a message needs mid-depth features, not new weights

Classifying a single message is the easiest case for appended modules. Frozen features lose the most on tasks that need new interactions between two texts. For BERT, fine-tuning beats frozen features by **2.3–6.7 points on sentence-pair tasks, but only about 0.5 points on single-sentence tasks** such as sentiment and NER ([Peters et al. 2019](https://ar5iv.labs.arxiv.org/html/1903.05987)). A Slack message or support ticket is a single text with a label. The in-distribution gap between a linear probe on frozen features and full fine-tuning is small. On mostly vision benchmarks, fine-tuning obtains about **2% higher accuracy in-distribution but 7% lower accuracy out-of-distribution**. The cause is that updating lower layers distorts pretrained features ([Kumar et al. 2022](https://arxiv.org/abs/2202.10054)). Slack traffic drifts as teams, products and phrasing change, so that robustness is a real advantage, not a consolation prize.

Recent decoder-LLM studies confirm this. In a 2025–26 comparison of approaches:
- A **frozen LLM2Vec encoder plus a head scored the best F1 (0.833)** on a 14-class multi-label patent task, ahead of the LoRA and instruction-tuned variants.
- On a 5-class task, a LoRA-adapted Llama-3.2-3B with a classification head reached **F1 0.860 with 12.2M trainable parameters**. Fully fine-tuned PatentBERT reached 0.854 with 346M.
- Zero-shot prompting of the same 3B model scored **0.21**.

The resulting classifier was calibrated well enough (ECE 0.094) for confidence-based routing, and distilling it into ModernBERT-base kept **97% of F1 at about 45× the speed** ([Yousefiramandi & Cooney 2025](https://arxiv.org/html/2512.12677)). Across 17 tasks, penalized logistic regression on a small LLM's embeddings "equals (and usually betters)" a large prompted LLM ([Buckmann & Hill 2024](https://arxiv.org/abs/2408.03414)).

**Which layer the head reads from matters.** Across 32 MTEB tasks, intermediate layers beat the final layer by **2% to 16% on average**, with the best layer usually near mid-depth ([Skean et al. 2025](https://arxiv.org/abs/2502.02013)). One team applied this to Qwen 2.5 0.5B–3B. They cut the model at its best intermediate layer and trained logistic regression on that layer's hidden state. With **fewer than 100 examples it beat GPT-4o** on content-safety and prompt-injection classification ([Sawtell et al. 2024](https://arxiv.org/abs/2412.13435)). This is where appending a head and saving compute meet. Truncating a 32-layer model at mid-depth roughly halves the cost of processing each message, and it improves accuracy. (The halving is simple arithmetic from layer count, not a measured figure.)

Static truncation also beats dynamic early exit. Newer LLMs have less redundancy between layers, which reduces the benefit of exiting early token by token ([Wei et al. 2026](https://arxiv.org/abs/2603.23701)).

Off-the-shelf embedding models make this even simpler. MTEB's classification benchmark *is* the "frozen embedder plus logistic regression" protocol ([Muennighoff et al. 2022](https://arxiv.org/abs/2210.07316)). On it, **Qwen3-Embedding-8B scores 90.43 and the 0.6B model 85.76** in English. On multilingual classification the 8B scores 74.00, while English-focused NV-Embed-v2 drops to 57.29 ([Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)).

When labels are scarce or routes change often, lighter methods are nearly as good.
- **SetFit:** with **8 examples per class**, it matched RoBERTa-large fine-tuned on 3,000 examples, and trained in 30 seconds for about $0.025 ([Hugging Face 2022](https://huggingface.co/blog/setfit)).
- **Centroids:** on 8,968 real IT-service tickets across 123 categories, a classifier built from embedding centroids matched an SVM (hierarchical F1 **0.731 vs. 0.727**). It updated incrementally **152× faster than retraining** ([Mohanna & Ait-Bachir 2025](https://arxiv.org/abs/2511.20667)). That speed is what lets you add a new Slack channel without a training job.

The comparison with prompting a large LLM is clear. On a narrow, stable label set, trained classifiers win; on broad or changing ones, the LLM holds its own.
- On ATIS, fine-tuned RoBERTa scored **95.9% against 84.1%** for zero-shot Claude Haiku.
- On the 150-intent CLINC150, they tied (**89.1% vs. 88.5%**, not significant).
- The LLM detected out-of-scope queries far better (**F1 85.0% vs. 73.0%**).
- A classifier locked to one app's schema scored **0% on an unseen app**, while the schema-prompted LLM held 93.9%.
- RoBERTa ran at **2.4 ms p50, against 981 ms** for the LLM ([Rodrigues & Vas 2026](https://arxiv.org/html/2608.20371)).

Small open models used zero-shot are no substitute. Across 41 models up to 9B, average zero-shot accuracy on BANKING77 was **0.388** ([Ganesh et al. 2026](https://arxiv.org/html/2607.27421)).

These results point to one design:
1. Run one frozen embedder, or an LLM cut off at mid-depth, once per message.
2. Cache the vector.
3. Attach cheap heads for team, ticket category and urgency.
4. Add routes by refitting a centroid or logistic-regression head.
5. Send low-confidence or out-of-scope messages to an LLM.

If the gap to full fine-tuning matters for a given head, LoRA with an embedding head closes it at a small parameter cost.

## A thousand task heads cost less than one extra model copy

**Storage** falls by three to six orders of magnitude once the backbone is shared:

| Per-task artifact | Size per task | 1,000 tasks, including base |
|---|---|---|
| Full fine-tuned Mistral-7B copy | 14.48 GB ([HF](https://huggingface.co/blog/multi-lora-serving)) | ≈14.5 TB |
| LoRA adapter on 7B | 13.6 MB ([HF](https://huggingface.co/blog/multi-lora-serving)) | ≈28 GB |
| MLP moderation probe on one layer | ≈8 MB ([arXiv 2606.10487](https://arxiv.org/html/2606.10487)) | ≈22.5 GB |
| Linear head, 4,096-d input, 10 classes | ≈80 KB (arithmetic) | ≈14.6 GB |
| Soft prompt on T5-XXL (5 tokens) | 20,480 params vs. 11B per copy ([Lester et al.](https://arxiv.org/abs/2104.08691)) | — |

At GPT-3 scale the same arithmetic gives **354 GB for 100 LoRA-adapted models against 35 TB for 100 full fine-tunes** ([Hu et al. 2021](https://ar5iv.labs.arxiv.org/html/2106.09685)). Adapter size depends on rank and on which modules are adapted. Adapting every linear layer, as current best practice recommends ([Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)), makes adapters several times larger, but still orders of magnitude smaller than copies.

**Training compute depends on whether gradients must flow back through the backbone.**
- **Inside-the-network methods** (LoRA, adapters and prefix tuning) still need a full backward pass through every layer above the lowest adapted one. They skip weight gradients and optimizer states, which brings compute to about **two-thirds of full fine-tuning per pass** ([Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)). Activation memory remains, so adapters and LoRA save only about **26% of training memory on T5-base**.
- **Ladder Side-Tuning,** which never backpropagates through the backbone, saves **69%** ([Sung et al. 2022](https://ar5iv.labs.arxiv.org/html/2206.06522)).
- **Heads on cached features** go furthest. Recent probe papers pre-extract hidden states from frozen LLMs, cache them, and train probes offline ([arXiv 2601.13288](https://arxiv.org/html/2601.13288)). The backbone then runs one forward pass per training example, however many heads or epochs follow. By standard FLOP accounting (about 6N FLOPs per token to train, 2N for a forward pass), ten tasks trained for three epochs each use roughly **90× less backbone compute** than ten separate fine-tunes. This figure is arithmetic, not a measured benchmark.

**Serving** cost per extra task is close to zero.
- S-LoRA serves **2,000 adapters on a single GPU** with up to 4× the throughput of a vLLM baseline and 30× that of Hugging Face PEFT ([Sheng et al. 2023](https://ar5iv.labs.arxiv.org/html/2311.03285)).
- Punica batches requests for different adapters at **12× throughput, adding 2 ms per token** ([Chen et al. 2023](https://arxiv.org/abs/2310.18547)).
- Loading 30 adapters in TGI raised VRAM by 3% ([Hugging Face 2024](https://huggingface.co/blog/multi-lora-serving)).

Heads on a shared forward pass are cheaper still. An 8 MB streaming-moderation probe used **0.6 GFLOP per request against 8.02 TFLOP for a separate 8B guard model**, at **0.1 ms against 448 ms** p50 latency ([arXiv 2606.10487](https://arxiv.org/html/2606.10487)). A 35M-parameter attention probe on Llama-3.2-3B beat a standalone ToxicChat-T5 classifier (**84.51 vs. 82.2 F1**), added 13.8 ms rather than the 88–123 ms of a separate guard pipeline, and peaked at 6.9 GB of GPU memory against 22.8 GB ([arXiv 2601.13288](https://arxiv.org/html/2601.13288)).

The pattern across these methods is a trade-off between expressive power and cost. The cheapest options are the least expressive, and routing happens to fall at the cheap end:

| Where new parameters live | Backprop through backbone? | Added inference cost | Fit for routing |
|---|---|---|---|
| Head on cached mid-depth features | No, one forward pass | ≈0.1 ms to ~14 ms | Strong for single-text labels |
| Side network with per-layer taps | No | Parallel side compute | About 1 point below FT |
| Soft prompt or prefix | Yes, all layers | Extra tokens or KV cache | Mature, batchable |
| Unmerged adapters or LoRA | Yes, above lowest adapter | +2–30% or ~2 ms/token | Closes last few points |
| New transformer blocks | Yes, above lowest block | +25% layers, always | Overkill; built for new domains |

In dollars, a February 2026 study found that fine-tuned BERT-class encoders on Cloud Run cost **$5–$33 per million requests**. Prompting GPT-4o or Claude Sonnet 4.5 cost **$192–$2,702**. The encoders matched or beat the LLMs on three of four benchmarks ([arXiv 2602.06370](https://arxiv.org/html/2602.06370v1)). Cheaper models have since narrowed the gap. Consider 1M messages a day at about 300 input tokens each:
- **gpt-5-nano:** about **$17 a day** at $0.05/$0.40 per million input/output tokens ([OpenAI](https://developers.openai.com/api/docs/pricing)).
- **Claude Haiku 4.5:** about $325 a day at $1/$5 ([Anthropic](https://platform.claude.com/docs/en/about-claude/pricing)).
- **CPU encoder:** about $5–11 a day.
- **One L4 GPU running a multi-LoRA 7B server:** a flat **$19 a day, whether it hosts 1 task or 30** ([Hugging Face 2024](https://huggingface.co/blog/multi-lora-serving)).

These daily figures are my own arithmetic from list prices. The main point is that each additional task multiplies an API bill but leaves the cost of a shared backbone unchanged.

## The savings leak through depth, hard tasks and base-model upgrades

**Added depth costs latency permanently.**
- Sequential adapters on GPT-2 medium added **20.7–30.3% latency at batch size 1**, shrinking to 2–3% at batch 32, because adapter layers "have to be processed sequentially" ([Hu et al. 2021](https://ar5iv.labs.arxiv.org/html/2106.09685)).
- Fusing eight adapters with AdapterFusion made inference **62% slower** ([Rücklé et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.11918)).
- LLaMA Pro runs 40 layers where its base ran 32, on every request ([Wu et al. 2024](https://arxiv.org/html/2401.02415)).
- Proxy-tuning's logit arithmetic slows decoding **1.5–2.4×** ([Liu et al. 2024](https://arxiv.org/html/2401.08565)).

Classification avoids part of this cost, for a reason that is easy to miss. PreFT found that unmerged multi-LoRA serving wastes **1.6–2.4× throughput** because decoding is memory-bound, and fixed it by applying adapters only to prefill tokens ([arXiv 2605.14217](https://arxiv.org/html/2605.14217)). A router emits one label, so it is almost pure prefill, and that decode penalty barely applies.

**Hard tasks reopen the gap.** LoRA "substantially underperforms full finetuning" on code and math, where full fine-tuning learns perturbations of 10–100× higher rank ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673)). The 8 MB moderation probe gave up **6 F1 points** to the 8B guard in exchange for its 10,000× compute saving ([arXiv 2606.10487](https://arxiv.org/html/2606.10487)). Closed-set heads also detect out-of-scope messages poorly, so an LLM fallback is required rather than optional ([Rodrigues & Vas 2026](https://arxiv.org/html/2608.20371)). Volume matters too. At 10,000 messages a day, gpt-5-nano costs cents while a dedicated GPU still costs $19. Self-hosting pays off only at hundreds of thousands to millions of requests a day, or when many tasks share the hardware.

**Base-model upgrades are the largest hidden cost.** Every appended module is fitted to one backbone's internal representations. Retraining a LoRA takes about **4 A100-hours, so upgrading 2,000 adapters costs roughly 8,000 GPU-hours** ([ReLoRA 2026](https://arxiv.org/html/2606.02606)). Heads on cached features have the same problem, because the cache must be recomputed. Hosted fine-tunes are pinned to a dated model snapshot. For OpenAI's gpt-4.1-nano and gpt-4o-mini, inference on a fine-tune costs about **twice the base per-token price** ([OpenAI](https://developers.openai.com/api/docs/pricing)).

The August 2026 UpgradeBench preprint, which followed Qwen 1.5 through Qwen 3, gives the most useful evidence here ([Chen & Zhang 2026](https://arxiv.org/html/2608.20918)):
- **Durability depends on the task.** Intent classifiers are "data-bound" and kept **99–101% of their gains across nine upgrade hops** when frozen on their original base. Text-to-SQL specialists, whose skill comes from the base model, lost up to 59% in a single hop.
- **Copying an adapter depends on training continuity, not architecture.** Moving an adapter to an architecturally identical but independently pretrained base failed: **Banking77 fell from 92.8% to 42.9%**. Moving it to a continued-pretraining release of the same model worked.
- **Relabeling works without gold labels.** Having the old specialist label retained traffic and training a new adapter from those labels reached **0.96–1.05 retention**.

The conclusion is that the weights are not the portable asset. **The labeled data, plus the old model acting as a teacher, is what survives an upgrade.**

## Six open problems where appended modules could be genuinely novel

Appending layers to a frozen model is a 2019 idea. LoRA, adapters and block expansion are standard practice. Novel work lies where gaps that papers themselves state overlap with classification on a shared backbone. The table lists six candidates. Each "why it is open" entry rests on a cited gap or on the absence of any result in the surveyed literature. The proposed experiments are my synthesis.

| Opportunity | Why it is open | First experiment |
|---|---|---|
| **1. Routers that survive upgrades** | UpgradeBench leaves learned mappings between models of different shapes, and data-free refresh, unmeasured; no source tests whether probes, steering vectors or ReFT survive an upgrade ([UpgradeBench](https://arxiv.org/html/2608.20918)) | Train probe, DiffMean and rank-1 LoReFT routers on Qwen2; transfer them to Qwen2.5 and Qwen3 using anchor representations, CKA layer matching, or text prototypes re-encoded on the new base; compare with UpgradeBench's Freeze, Copy and Refresh baselines on Banking77, CLINC150 and a Slack set |
| **2. Taxonomy-to-classifier hypernetworks** | Text-to-LoRA, Doc-to-LoRA, SHINE and HyperSteer all target generation or QA; none emits a classifier ([T2L](https://arxiv.org/abs/2506.06105), [D2L](https://pub.sakana.ai/doc-to-lora/), [HyperSteer](https://arxiv.org/abs/2506.03292)) | Input channel names, descriptions and k examples; output a linear head or rank-1 intervention in one pass; evaluate zero-shot on held-out taxonomies against SetFit, centroids and a schema-prompted LLM |
| **3. Many heads from one forward pass** | No published cost curve for N heads on one cached backbone; no benchmark of multiple interventions in one pass | Share one prefill, then run heads via aLoRA-style late activation ([aLoRA](https://arxiv.org/abs/2504.12397)), position-local ReFT, and mid-layer probes; measure interference, accuracy and throughput from N=1 to N=1,000 |
| **4. RL-trained tiny classifiers** | TinyLoRA's 13-parameter result was shown only on math, and only with RL; with supervised fine-tuning (SFT) it needed 100–1000× larger updates ([TinyLoRA](https://arxiv.org/abs/2602.04118v1)) | For a K-way router, compare cross-entropy with a correctness reward on 10- to 1,000-parameter adapters; measure sample efficiency, calibration and robustness to label noise |
| **5. Adding routes without forgetting** | Sparse memory fine-tuning was tested on facts, not routes; "when the number of skills is large, DATA-MIX is still the best" ([Lin et al.](https://arxiv.org/abs/2510.15103), [LoRA Soups](https://arxiv.org/html/2410.13025v1)) | Add routes through sparse memory slots or a library of I2CL-style task vectors; measure forgetting of old routes against LoRA and against refitting the head |
| **6. Top-appended block vs. adapters spread through depth** | No controlled study on LLM-era decoders at a matched parameter budget; LLaMA Pro reported only general scores for top stacking ([Wu et al.](https://arxiv.org/html/2401.02415)) | Truncate at the best mid-layer, append 1–2 identity-initialized blocks, and compare with adapters, LoRA and a linear probe on routing data, measuring both accuracy and latency |

**Opportunity 1 is the strongest bet for this use case.**
- **The failure is well documented:** naive adapter copying drops Banking77 from 92.8% to 42.9%.
- **The cost is measured:** thousands of GPU-hours per upgrade cycle.
- **There is evidence it is solvable.** Classifiers built on SAE features transferred from Gemma 2 2B to Gemma 2 9B-IT ([Gallifant et al. 2025](https://aclanthology.org/2025.emnlp-main.1521/)). Memory Decoder plugs into any model that shares its tokenizer ([arXiv 2508.09874](https://arxiv.org/abs/2508.09874)).

Prototype heads offer a hint. A centroid built from text examples is portable by construction: you re-embed the examples on the new backbone. The open research question is how much of LoRA-level accuracy can be recovered from a head that is fully re-derivable from text.

**Opportunity 2 complements it.** Schema-prompted LLMs keep 93.9% accuracy on unseen route sets but are about 400× slower than a trained classifier. A hypernetwork that emits a head from a taxonomy description would combine the LLM's flexibility with a classifier's speed.

**Opportunity 4 is the cheapest experiment and the most interesting scientifically.** A label carries about log₂K bits, which is information-light in the same way an RL reward is. Classification is therefore the natural test of whether TinyLoRA's result reflects something about RL, or about supervision that carries little information.

Three cautions apply. First, activation steering, which uses the same activation-space machinery as opportunities 1 and 5, is unreliable outside the distribution it was built on. Its effects "vary significantly across behaviors, and are often unreliable or even counterproductive" ([arXiv 2504.04635](https://arxiv.org/pdf/2504.04635)). Second, SAE-feature probes rarely beat logistic regression on raw activations ([Kantamneni et al. 2025](https://arxiv.org/pdf/2502.16681)). Any interpretable router needs strong baselines. Third, these novelty judgments cover only literature surfaced through September 2026. Several very recent titles on composing steering vectors and hard-routed LoRA mixtures were seen but not read. Do a targeted literature check before committing to any direction.

## Conclusion

Asking whether to append layers or adjust weights is less useful than asking whether a task needs to change the model's internal computation, or only to read from it. Single-message routing needs only to read. That is why a head on frozen mid-depth features gets within a few points of fine-tuning, holds up better under drift, and lets a thousand routers share the storage footprint of about two model copies. Routing classifiers also sit in an unusual spot. They are information-light enough that tiny additive modules suffice, and data-bound enough that a frozen specialist keeps its gains across upgrades. That makes a routing team's production problem a good testbed for questions the wider field has not answered.

For deployment, treat the labeled examples plus a teacher model as the durable asset, and the appended weights as a disposable, per-backbone artifact. The genuinely novel direction for a small team is not another way to attach parameters. It is making appended modules portable, generated on demand, and servable by the thousand from a single forward pass.
