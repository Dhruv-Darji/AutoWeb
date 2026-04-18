# Literature Review: DOM Delta Processing for LLM-Based Web Agents

## 2.1 DOM Representation and Reduction for Web Agents

Large Language Model (LLM)-based web agents interact with websites by
consuming a representation of the Document Object Model (DOM). However,
real-world web pages routinely contain 10,000--100,000 DOM tokens (Zhang
et al., 2025), far exceeding the practical context limits of current LLMs
and degrading both cost and reasoning quality. Consequently, a significant
body of work has focused on reducing the DOM representation before it
reaches the LLM.

**Mind2Web** (Deng et al., 2023) introduced a two-stage architecture that
has become the de facto standard. In Stage 1, a fine-tuned DeBERTa-v3
cross-encoder ranks all interactive elements against the task instruction
and returns the top-K candidates. In Stage 2, a large LLM (GPT-4) selects
the target element from this reduced set. The dataset provides stable
element identity through Chrome DevTools Protocol `backend_node_id`,
enabling reliable cross-render element tracking. However, Mind2Web
processes each action step independently --- there is no state carried
between steps, meaning the full candidate pool is re-ranked from scratch at
every step regardless of how little the page changed.

**AutoWebGLM** (Lai et al., 2024) addresses DOM size through an HTML
Pruner algorithm that iteratively simplifies the DOM tree by removing
non-actionable nodes, limiting tree depth and sibling count, and
preserving nodes with actionable descendants. The observation space
includes simplified HTML, task description, viewport position, and
previous action history. While action history is tracked in text, the
DOM representation itself is stateless --- each step re-simplifies the
full current DOM independently.

**D2Snap** (Schiepanski et al., 2025) is the first algorithm specifically
designed for DOM downsampling. It operates via post-order tree traversal
with three operations: (1) merging container elements hierarchically
(parameter k), (2) dropping low-relevance sentences using TextRank
(parameter l), and (3) filtering attributes below a semantic threshold
derived from GPT-4o ground truth (parameter m). D2Snap reduces DOM size
to ~10^3 tokens, matching the token order of grounded GUI screenshots.
An adaptive wrapper (AdaptiveD2Snap) auto-tunes parameters to hit a
target token budget. However, D2Snap operates on static snapshots in
isolation --- it contains no mechanism for tracking state changes across
consecutive agent steps.

**Prune4Web** (Zhang et al., 2025) proposes a meta-programming approach:
the LLM generates executable Python scoring scripts that dynamically
filter DOM elements based on semantic cues from decomposed sub-tasks.
This achieves 25x--50x reduction in candidate elements and improves
grounding accuracy from 46.8% to 88.28%. The approach is task-aware but
requires an additional LLM call to generate the pruning script, adding
cost and latency at the filter stage.

**Browser-Use** (2024--2025) implements a four-stage pipeline that fuses
DOM structure, accessibility tree semantics, and visual snapshot data.
Stage 1 executes parallel Chrome DevTools Protocol requests; Stage 2
creates unified `EnhancedDOMTreeNode` structures; Stage 3 applies
visibility and interactivity filtering; Stage 4 serializes for the LLM.
This reduces ~10,000 DOM nodes to ~200 interactive elements. The pipeline
is highly effective per-step but entirely stateless across steps.

A common limitation unites these approaches: **all process each DOM
snapshot independently, without leveraging the temporal relationship
between consecutive page states.** When a web agent clicks a button and
only a dropdown menu appears, these systems re-process the entire page
rather than surfacing only what changed.


## 2.2 DOM Change Tracking in Web Agents

The only published system that explicitly monitors DOM changes between
agent steps is **Agent-E** (Tarsariya et al., 2024). Agent-E introduces
a "change observer" module that leverages the browser's MutationObserver
Web API to capture DOM mutations in real time immediately after action
execution. The observer monitors attribute changes (e.g., `aria-expanded`)
and new DOM nodes, then generates a natural-language description of the
mutations (e.g., "Clicked the element with mmid 25. As a consequence, a
popup has appeared with the following elements"). This linguistic feedback
is appended to the LLM's next prompt to help the agent verify action
outcomes and plan subsequent steps.

Agent-E's change observer differs from DOM delta processing in three
fundamental ways. First, **mechanism**: it uses real-time browser-level
mutation events, not parsed snapshot comparison. Second, **purpose**: it
serves as an action verification and feedback signal ("did the click
work?"), not as a candidate pool reduction pipeline. Third, **output**:
it adds natural-language text to the prompt, increasing token count rather
than reducing it. Agent-E achieves 73.2% success rate on WebVoyager, but
the change observer's contribution to token efficiency is not measured.

**Agentic Compilation** (2025) mentions "DOM diffing, prompt compression,
and partial state caching" as production-oriented mitigations in its
Related Work section, but does not implement DOM diffing. Its own
contribution, a DOM Sanitization Module (DSM), achieves up to 85% token
compression through per-step noise removal and attribute cleansing.


## 2.3 Semantic Relevance Scoring for DOM Elements

The standard approach to scoring DOM elements against task instructions is
keyword or pattern matching. Mind2Web uses a fine-tuned cross-encoder;
AutoWebGLM uses structural heuristics; Browser-Use uses accessibility tree
roles. None of these systems apply sentence-level semantic embeddings for
element relevance scoring.

**Sentence-BERT** (Reimers and Gurevych, 2019) established the methodology
for encoding sentences into dense vectors via siamese/triplet BERT
networks, enabling efficient cosine-similarity-based retrieval. The
late-fusion pattern --- combining sparse (keyword/BM25) scores with dense
(embedding) scores --- has been shown to outperform either component alone
in information retrieval benchmarks.

**MiniLM** (Wang et al., 2020) distills large Transformer encoders into
compact models with minimal performance loss. The `all-MiniLM-L6-v2`
checkpoint (6 layers, 22M parameters, ~80 MB) ranks near the top of
sentence similarity benchmarks for its size class while running at ~2 ms
per sentence on CPU, making it suitable for latency-sensitive pipelines.

No existing web agent system uses SBERT-style embeddings for DOM element
relevance scoring. Web agents either rely on task-specific fine-tuned
rankers (Mind2Web's DeBERTa), structural heuristics (AutoWebGLM), LLM-
generated scripts (Prune4Web, Zhang et al.), or send raw DOM text to the
LLM without intermediate scoring (SeeAct). This leaves a gap: paraphrase
pairs that are routine on the web (task: "purchase" vs. button:
"Checkout"; task: "sign in" vs. link: "Log in") are missed by keyword
matching and not addressed by structural pruning.


## 2.4 Hierarchical Context for Element Disambiguation

**WebAgent / HTML-T5** (Gur et al., 2024) demonstrates that encoding the
ancestor chain of each DOM element is necessary for grounding. On pages
containing visually identical elements (e.g., an "email" input inside a
login form vs. a newsletter form), the element's text alone is
insufficient --- its position within the DOM hierarchy provides the
distinguishing signal. HTML-T5 embeds this hierarchical context by
fine-tuning a T5 model on full HTML token sequences, an approach that is
effective but model-specific and training-intensive.

**SeeAct** (Zheng et al., 2024) sidesteps the text representation problem
by grounding through screenshots annotated with element markers, using
GPT-4V for multimodal reasoning. While effective for visual grounding,
this approach cannot leverage structural DOM information and is dependent
on the quality of visual annotations.


## 2.5 Research Gap

The literature reveals a clear gap at the intersection of three
capabilities:

1. **Cross-step DOM state tracking** --- comparing consecutive page
   snapshots to identify what changed, rather than re-processing the full
   DOM from scratch. Agent-E's change observer is the only system that
   approaches this, but it serves a different purpose (action feedback, not
   candidate filtering) and uses a different mechanism (real-time browser
   events, not snapshot comparison).

2. **Task-aware semantic relevance scoring** --- using sentence embeddings
   to match task descriptions against DOM element attributes, capturing
   paraphrases that keyword matching misses. No existing web agent applies
   SBERT-style dense retrieval to DOM elements.

3. **Lightweight hierarchical context** --- providing ancestor-chain
   information to the grounder without requiring model fine-tuning, as a
   model-agnostic text suffix rather than an embedded representation.

Our DOM Delta Processing contribution addresses this gap by combining
snapshot-to-snapshot diffing with hybrid keyword + embedding relevance
scoring and ancestor-augmented element summaries.


## References

1. Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang, B., Sun,
   H., and Su, Y. (2023). Mind2Web: Towards a Generalist Agent for the
   Web. *Advances in Neural Information Processing Systems (NeurIPS)*,
   36. arXiv:2306.06070.

2. Lai, H., Liu, Z., Mao, J., Feng, J., Chen, Y., Xia, Y., Wang, Z.,
   and Tang, J. (2024). AutoWebGLM: A Large Language Model-based Web
   Navigating Agent. *Proceedings of the 30th ACM SIGKDD Conference on
   Knowledge Discovery and Data Mining*. arXiv:2404.03648.

3. Schiepanski, T. et al. (2025). Beyond Pixels: Exploring DOM
   Downsampling for LLM-Based Web Agents. arXiv:2508.04412.

4. Zhang, J., Chen, K., Lu, Z., Zhou, E., Yu, Q., and Zhang, J. (2025).
   Prune4Web: DOM Tree Pruning Programming for Web Agent.
   arXiv:2511.21398.

5. Tarsariya, A. et al. (2024). Agent-E: From Autonomous Web Navigation
   to Foundational Design Principles in Agentic Systems.
   arXiv:2407.13032.

6. Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
   using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*.
   arXiv:1908.10084.

7. Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. (2020).
   MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression
   of Pre-Trained Transformers. *Advances in Neural Information Processing
   Systems (NeurIPS)*, 33. arXiv:2002.10957.

8. Gur, I., Furuta, H., Huang, A., Saber, M., Matsuo, Y., Eck, D., and
   Fishi, A. (2024). A Real-World WebAgent with Planning, Long Context
   Understanding, and Program Synthesis. *International Conference on
   Learning Representations (ICLR)*. arXiv:2307.12856.

9. Zheng, B., Gou, B., Kil, J., Sun, H., and Su, Y. (2024). GPT-4V(ision)
   is a Generalist Web Agent, if Grounded. *International Conference on
   Machine Learning (ICML)*. arXiv:2401.01614.

10. Agentic Compilation (2025). Mitigating the LLM Rerun Crisis for
    Minimized-Inference-Cost Web Automation. arXiv:2604.09718.
