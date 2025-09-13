# Implementing **Mixture-of-Recursions (MoR)** in a Retrieval-Augmented Customer-Support Bot — Research-level README

> Short summary — this document explains the theory (Transformer / *Attention Is All You Need*), the MoR paper (what it introduces and why it helps), and a practical research-grade plan to implement a **RAG (retrieval-augmented generation)** customer-support bot stack using **ChromaDB + Streamlit + (optionally) TinyLLM for on-device/small-model embeddings**, with concrete engineering notes, evaluation plans, and links to code and papers.

---

# 1. High-level motivation

Customer-support bots must give accurate answers grounded in product docs, KBs and policy while being cost-efficient and fast. Two orthogonal levers matter:

* **Retrieval (RAG)**: keep a small, curated document store and retrieve grounding context (cheap, interpretable).
* **Model efficiency**: LLMs deliver quality but are costly. Architectures that give *quality per compute* improvements allow deploying stronger behaviour for the same budget.

**MoR** (Mixture-of-Recursions) is a recent Transformer variant that unifies parameter sharing and token-level adaptive compute to get better quality/throughput tradeoffs than standard Transformers, making it attractive for production RAG systems where latency and cost matter.&#x20;

---

# 2. Quick theoretical background (Transformer — *Attention Is All You Need*)

Key ideas you must carry into system design:

* **Self-attention / Scaled dot-product**: each token computes attention weights with all keys, producing constant-depth connections between any pair of tokens; attention formula: `softmax(QKᵀ / sqrt(dk)) V`. This is the building block.&#x20;
* **Multi-head attention**: parallel attention heads let the model attend to multiple subspaces. This improves expressivity versus a single attention.&#x20;
* **Positional encoding**: since Transformers drop recurrence/convolution, positional signals (sin/cos or learned embeddings) are added to preserve order.&#x20;
* **Why Transformers helped**: they enable massive parallelism, shorter path lengths for long-range dependencies, and excellent empirical results on translation and many downstream tasks.&#x20;

If you need a compact primer to cite in experiments, see the original paper.&#x20;

---

# 3. What MoR brings (core contributions & intuition)

**Mixture-of-Recursions (MoR)** unifies three efficiency axes inside a **Recursive Transformer**:

1. **Parameter sharing (recursive layers)** — reuse the same stack of layers multiple times (reduces unique parameter count).
2. **Token-level adaptive recursion (routers / early exit)** — a lightweight router decides how many recursion steps (how many times the shared block is applied) each token needs, so “hard” tokens get more compute than function words.&#x20;
3. **KV caching strategies** — selectively cache key/value pairs recursion-wise or share KV from the first recursion to reduce KV memory and IO during autoregressive decoding.&#x20;

Key empirical points from the paper (summary):

* Under equal training FLOPs, MoR achieves *better validation perplexity and few-shot accuracy* than vanilla and recursive baselines while using fewer unique parameters — i.e., a new Pareto frontier.&#x20;
* Two routing paradigms are studied: **expert-choice** (top-k per recursion depth) and **token-choice** (assign a token an expert/depth at the start); each has tradeoffs (expert-choice can leak information / needs mitigation; token-choice needs load balancing).&#x20;
* **KV caching**: recursion-wise caching reduces KV memory and IO for deeper layers; recursive KV-sharing reduces prefill latency but can slightly hurt accuracy depending on routing.&#x20;

For the rigorous write-up and ablation tables, consult the MoR paper and the authors' repo.  ([GitHub][1])

---

# 4. Why MoR can be *better than a standard Transformer* for a RAG support bot

Practical advantages for a customer-support RAG system:

* **Adaptive compute for relevance**: tokens in the retrieved context or user query that require deeper reasoning can receive more recursion steps; routine tokens exit early — saves inference compute. (MoR is token-adaptive).&#x20;
* **Parameter efficiency**: sharing reduces model size for the same effective depth — cheaper to host (especially attractive for on-prem or edge deployments).&#x20;
* **KV-aware decoding for throughput**: recursion-wise caching reduces KV IO and can increase decoding throughput — useful for many concurrent queries.&#x20;

Caveats: MoR requires extra engineering (routing modules, KV caching, balancing load) and careful hyperparameter tuning (Nr, capacity/top-k, caching mode) — see the paper's ablations.&#x20;

---

# 5. System architecture (proposed): MoR + RAG customer support

High-level components:

1. **Document ingestion & chunking**

   * Source: product docs, policies, KB, past tickets.
   * Chunk size: 200–800 tokens; overlap 50–100 tokens for context continuity.
   * Preprocess: normalize, remove PII, map metadata (doc id, section).

2. **Embeddings / vectorization**

   * Option A (recommended for speed & compatibility): use a proven embedding model (e.g., `all-MiniLM-L6-v2` or other open models) to create dense vectors; store in ChromaDB. (see embedding model guides). ([BentoML][2])
   * Option B (edge / custom): use **TinyLLM** if you want an edge-fitted small LLM or to train a tiny embedding head tailored to your KB. TinyLLM is a framework for training/deploying small LMs on constrained hardware; leveraging its embedding layer for vectors is possible but typically requires more engineering than using off-the-shelf embedding models. (see TinyLLM resources). ([arXiv][3])

3. **Vector store / Retriever**

   * **ChromaDB** to store embeddings and run k-NN (approximate / exact) retrieval; returns top\_k grounded chunks and metadata. Example RAG tutorials show integration with Streamlit and Chroma. ([Michael Scheiwiller][4])

4. **Generator / LLM**

   * Two paths:

     * (a) **MoR-based generator**: research route — fine-tune or pretrain a MoR model (or adapt its codebase) and run it as the generator. This gives the potential MoR efficiency gains but requires access to MoR codebase and compute to pretrain or up-train from a checkpoint.  ([GitHub][1])
     * (b) **Standard LLM generator** (fast MVP): use a Llama-family model, OpenAI, or local LLM; prompt with the retrieved chunks and produce final answer. This is easiest to get working and can be replaced by MoR later.

5. **Application UI**

   * **Streamlit** front-end for developer dashboard, conversation UI, and diagnostics (retrieved chunks, embeddings visualizer). Several blog posts show Streamlit + Chroma prototypes. ([Michael Scheiwiller][4])

6. **Safety, grounding & caching**

   * Always include retrieval provenance in responses, show top-k sources, enable “show retrieval” toggle in UI.
   * Rate-limit model generation, keep a fallback policy response if retrieval confidence is low.

---

# 6. Implementation roadmap (research → prototype → deploy)

### Phase 0 — Quick prototype (MVP)

* Use existing embeddings (sentence-transformers) → ChromaDB → LLM (OpenAI or local Llama) generator → Streamlit UI. Follow standard RAG tutorials. ([Medium][5])

Deliverables:

* Working QA UI, retrieval breakdown, metrics logging.

### Phase 1 — Instrumentation & evaluation

* Add telemetry: latency, tokens/sec, retrieval precision\@k, grounding accuracy (human labels), hallucination rate.
* Baseline metrics: perplexity on KB QA set, answer F1, retrieval R-precision, average latency.

### Phase 2 — MoR research integration (experimental)

* Option A: Run MoR codebase on small scale (use their repo and configs) to reproduce paper baselines for a smaller domain corpus. Repo available. ([GitHub][1])
* Option B: If training is infeasible, implement MoR *ideas* as patches to your generator:

  * Add **early-exit logic** for token groups (coarse version): after some transformer blocks, apply a router to decide whether to run extra blocks for particular tokens (this approximates MoR). See MoR router designs for guidance (expert-choice vs token-choice).&#x20;
* Evaluate: isoFLOPs comparison vs baseline; measure throughput and accuracy tradeoffs as in MoR paper (validation NLL, few-shot accuracy proxies).

### Phase 3 — Deploy & optimize

* Choose caching strategy: recursion-wise caching is safer for accuracy; recursive KV sharing helps latency but may degrade accuracy in some routing modes. Use MoR ablations to guide choice.&#x20;
* Use continuous depth-wise batching and efficient KV IO to maximize throughput in production.&#x20;

---

# 7. Concrete experimental plan (suggested)

1. **Baselines**:

   * Vanilla Transformer generator (Llama-style) + sentence-transformer embeddings + Chroma.
   * Metrics: retrieval precision\@k, answer accuracy (human), latency per query, tokens/sec.

2. **MoR experiments**:

   * Reproduce a small MoR model (e.g., 135M / 360M scale as in paper) trained on a domain subset; measure validation NLL and throughput vs vanilla of similar FLOPs.&#x20;
   * Ablations: expert-choice vs token-choice routing; recursion-wise caching vs recursive sharing; parameter-sharing strategy (Middle-Cycle works best in paper).

3. **RAG specific tests**:

   * Measure how often the MoR generator uses deeper recursion on tokens coming from retrieved context vs query tokens; inspect router outputs to interpret where compute goes. (MoR router analysis is shown in the paper.)&#x20;

4. **Operational tests**:

   * Throughput test with many concurrent users; measure KV IO and effective tokens/sec (paper reports 1.2–2× speedups for certain MoR configs under depth-wise batching).&#x20;

---

# 8. Engineering notes & best practices

* **Embeddings**: For most RAG flows use well-tested embedding models; only use TinyLLM embedding layers if you require a custom tiny model on edge and are willing to adapt/training. TinyLLM is a framework aiming at edge small LMs — useful if you plan to *train* an embedding head for your KB. ([arXiv][3])
* **ChromaDB**: easy to self-host and integrates with many frameworks; good for prototyping with Streamlit. ([Michael Scheiwiller][4])
* **Streamlit**: build the developer UI with retrieval debug panels (show chunks, scores, embedding distances). See community examples and “RAGxplorer” for diagnosing embeddings. ([Streamlit][6])
* **MoR repo**: start from the authors’ codebase to reproduce experiments and adapt their router/KV caching modules. ([GitHub][1])
* **Safety**: log sources with each answer, implement hallucination detectors (e.g., if retrieval score low, return “I’m not sure — here are sources” pattern).

---

# 9. Important links & resources

* **MoR (paper PDF, authors’ experimental details & design choices)** — local upload / canonical: (paper contents used above).&#x20;
* **MoR — supplementary / design & ablations (sections on routing / caching)**. &#x20;
* **MoR code (GitHub)** — authors' repo: raymin0223/mixture\_of\_recursions. ([GitHub][1])
* **Transformer — *Attention Is All You Need*** — seminal Transformer paper (attention, multi-head attention, positional encodings).&#x20;
* **TinyLLM (paper / project info)** — TinyLLM framework / arXiv: training & deploying very small LMs for edge; useful if you plan to train a small local embedding/generator. ([arXiv][3])
* **ChromaDB + Streamlit RAG examples / tutorials** — practical how-tos and demos for building local RAG apps. ([Michael Scheiwiller][4])
* **Embedding-model guides** (choosing an open embedding model): surveys and blog posts. ([BentoML][2])

---

# 10. Suggested next steps (practical)

1. **MVP (1–2 days)**: Build Streamlit + Chroma + off-the-shelf embeddings + generator pipeline. Log retrievals + answers. Use tutorials above. ([Michael Scheiwiller][4])
2. **Prototype MoR (1–3 weeks)**: Clone MoR repo, run small-scale training (use their configs for 135M / 360M if you have 4x GPUs), reproduce a simple isoFLOP comparison on your domain data. ([GitHub][1])
3. **Integrate & test**: Replace generator with MoR model when you have a usable checkpoint; run RAG queries and compare latency / accuracy. Measure routing behaviour on user queries (interpretability win).&#x20;

---

# 11. Quick FAQs

* **Can I use TinyLLM as the embedding generator out of the box?**
  TinyLLM is primarily a framework for small LLMs (training/deploy at edge). You *can* extract an embedding head or train a model to produce embeddings, but for quick RAG pipelines using standard embedding models (sentence-transformers, open embedding models) is faster and better tested. If your requirement is extreme edge / privacy, TinyLLM is worth exploring. ([arXiv][3])

* **Which MoR routing should I try first?**
  Start with **expert-choice** if you need stable load balancing and you can mitigate training leakage with auxiliary losses; try **token-choice** if you want non-leaky per-token decisions but be ready to add balancing loss / biasing to avoid imbalance. The paper compares both; see Table/Section on routing.&#x20;

* **Which KV caching strategy?**
  For best accuracy with selective routing: **recursion-wise caching**. If prefill latency / memory is the bottleneck and you can tolerate small accuracy tradeoffs, consider **recursive KV sharing**. Test both; the paper’s ablations guide this tradeoff.&#x20;

---

# 12. Closing / summary

MoR is a promising architectural innovation if you aim to squeeze more quality and throughput out of an LLM generator inside a RAG customer-support pipeline. The practical path is: (i) prototype RAG with standard components (ChromaDB + Streamlit + reliable embedding model), (ii) reproduce small MoR experiments from the authors’ repo, (iii) integrate MoR generator and evaluate cost/latency/accuracy tradeoffs on your support queries. The paper and repo give concrete design choices (routing types, caching) and ablations you can follow to iterate quickly.

---

If you want, I’ll:

* produce a **detailed experiment notebook** (data splits, training schedule, loss terms, Nr/hyperparameter grid) adapted to your compute; **or**
* generate a Streamlit + Chroma starter template (code) that you can run locally to prototype the RAG pipeline right now.

Which one do you want next?

[1]: https://github.com/raymin0223/mixture_of_recursions?utm_source=chatgpt.com "GitHub - raymin0223/mixture_of_recursions: Mixture-of-Recursions"
[2]: https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models?utm_source=chatgpt.com "A Guide to Open-Source Embedding Models - BentoML"
[3]: https://arxiv.org/abs/2412.15304?utm_source=chatgpt.com "TinyLLM: A Framework for Training and Deploying Language Models at the Edge Computers"
[4]: https://www.michaelscheiwiller.com/blog/legal-rag-streamlit-chromadb-openai?utm_source=chatgpt.com "Building a Streamlit-Powered RAG App with ChromaDB and OpenAI"
[5]: https://medium.com/%40piyushsonawane10/building-a-retrieval-augmented-generation-rag-application-with-streamlit-chromadb-and-c4544a621887?utm_source=chatgpt.com "Building a Retrieval-Augmented Generation (RAG) Application with ..."
[6]: https://discuss.streamlit.io/t/ragxplorer-explore-the-embeddings-of-your-rag-documents-gpt-4-chromadb-sentence-transformers/59371?utm_source=chatgpt.com "RAGxplorer - Explore the embeddings of your RAG Documents ..."
