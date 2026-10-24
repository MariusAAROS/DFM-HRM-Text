# graph_inject: fidelity review against the source papers

Review only: the reviewer changed no code in either pass.

- **Pass 1 (2026-10-24 morning)**: every method in [`graph_inject/methods/`](../../graph_inject/methods/) and
  [`graph_inject/configs/methods/`](../../graph_inject/configs/methods/), checked against its cited paper and the
  official code where it exists.
- **Pass 2 (2026-10-24, this version)**: re-check of the updated code. The working tree is uncommitted, on top of
  commit 19d7a65. This pass checks two things:
  - whether each pass-1 finding is fixed;
  - whether each fix, and each new arm (`graphtoken-faithful`, `node-tokens-faithful`), matches the **original
    paper** and not only the pass-1 note.

Sources (all read for this review):

| Method | Paper | Official code |
| --- | --- | --- |
| Less is More | arXiv 2508.10115 v1 | none |
| Talk like a Graph | ICLR 2024 | google-research encoders, as ported in GRL `graph_text_encoders.py` |
| Colorful Talks | arXiv 2602.10386 v2 | `angelozangari/CL-OWL` @ 1f22ea7 |
| P-Tuning | arXiv 2103.10385 | peft 0.21.2 |
| GraphToken | arXiv 2402.05862 v1 | none (TF-GNN) |
| Grover et al. | arXiv 2604.00342 v1 | (not checked) |
| LLaGA | arXiv 2402.08170 | `VITA-Group/LLaGA` @ 288240f |
| GTLM | arXiv 2605.10247 v1 | `DarioVajda/graph_model` (`src/models/bias.py` `SPDBias`, `src/experiments/graphqa/process_dataset.py`) |

Severity:
- **High**: the run does not test what its name or citation says.
- **Medium**: can move the numbers.
- **Low**: cosmetic, or a documented adaptation.

## Verification run in pass 2

- `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 pytest tests/test_graph_inject.py`: **33 passed, 48 skipped**. The GPU
  tests were skipped on purpose: the GPU was busy with the queue (`metaqa / hrm1b / rel-bias`). That includes
  `test_rel_bias_same_occurrence_is_zero` and the prefill/packed parity tests. **These are not re-verified here.**
  Run them once the queue frees the GPU.
- Spot checks on the real data, with throwaway scripts outside the repo:
  - **`graph_text: false` prompts.** These reduce to `#### Question\nQ: ...\nA: ` on GraphQA, MetaQA and KQA Pro:
    the task text only, as GraphToken's `T`.
  - **Query nodes (`question_nodes`, GraphQA test).** The count matches the task type exactly:
    - 0 nodes: CycleCheck, EdgeCount, NodeCount, TriangleCounting;
    - 1 node: ConnectedNodes, DisconnectedNodes, NodeDegree;
    - 2 nodes: EdgeExistence, MaximumFlow, Reachability, ShortestPath.
    - MetaQA: 1 mention in 95%.
    - KQA Pro: 0 mentions in 18% (questions that describe rather than name the entity).
  - **Answer string present in the question-only prompt.** MetaQA 0/1197, KQA Pro 4/1134. This confirms the
    documented "close to closed-book" caveat for the `-faithful` arms on the KGs.
  - **Max nodes per graph (all splits).** GraphQA 19, MetaQA 111, KQA Pro 180. All are below `idx_max: 256`, so
    `graphtoken-faithful` cannot hit its `ValueError`.
  - **`cl_owl_color` over [0, 1].** It reaches only `{red, crimson, orange, lime, green, teal}`. "lime" appears in
    two separate bands (≈0.45–0.5 and ≈0.72–0.78).

## Status of the pass-1 findings

| # | Pass-1 finding | Severity then | Status now |
| --- | --- | --- | --- |
| 1 | `text`: attributed to Less is More, which trains on instructive CoT | Medium | **Fixed (docs).** Now "the LoRA arm of Less is More, not its CoT recipe" (`text.py`, `text.yaml`, README). |
| 2 | `text-incident`: neighbour order unverified | Low | **Documented** as unverifiable (GraphQA incident files not on disk). Unchanged. |
| 3 | `text-colors`: "no public code" | Medium | **Fixed.** Cites `angelozangari/CL-OWL` @ 1f22ea7. |
| 4 | `text-colors`: 16-name palette | Medium | **Fixed.** The official `algos.py` naming is ported verbatim. |
| 5 | `text-colors`: template wording | Low | **Fixed.** Paper sentences verbatim (see §3). |
| 6 | `prompt-tuning`: ambiguous citation | Low | **Fixed.** "GPT Understands, Too", arXiv 2103.10385, MLP as in peft. Notes that it is Less is More's arm. |
| 7 | `graphtoken*`: graph text kept | High | **Fixed by a new arm.** `graphtoken` / `-frozen` are renamed to "GraphToken + text" (Less is More's arm). `graphtoken-faithful` drops the graph text. |
| 8 | `graphtoken*`: graph-level readout only | Medium | **Fixed in `-faithful`** (`readout: graph+query`). Unchanged in the "+ text" arms, as documented. |
| 9 | `graphtoken*`: no learned IDX features | Medium | **Fixed in `-faithful`** (`idx_dim: 16`). |
| 10 | "graph pooling tokens" (Grover et al.) claim | High | **Removed.** The label is gone from `graph_token.py`, both configs and the README. The README's deviation table records why it was wrong. No pooling-operator arm was added, so Grover et al. is not a baseline here. |
| 11 | `node-tokens`: 2 hops instead of HO's 4 | Medium | **Fixed by a new arm.** `node-tokens-faithful` uses `hops: 4`. `node-tokens` keeps 2, documented for phase-1 comparability. |
| 12 | `node-tokens`: LoRA instead of a frozen LLM | Medium | **Fixed in `-faithful`** (`regime: frozen`). |
| 13 | `node-tokens`: graph text kept | Medium | **Fixed in `-faithful`** (`graph_text: false`). |
| 14 | `rel-bias`: "no public GTLM code" | Medium | **Fixed.** Cites `DarioVajda/graph_model` @ f4feefb. |
| 15 | `rel-bias`: same-element bias learned (GTLM: b_SPD(u,u)=0) | High | **Fixed** (see §7). |
| 16 | `rel-bias`: text and sequential RoPE kept, no prompt node | High | **Re-scoped, documented.** Now "GTLM SPD bias on top of the text" (the RPE counterpart of `abs-pe`). It is no longer called GTLM. See §7. |
| 17 | `rel-bias`: SPD only | Medium | **Documented** as GTLM's SPD-only ablation arm. |
| 18 | `ToEmbeddingScale` deviation | Low/Medium | **Documented** in `base.py` and in the README deviation table. |

## New findings from pass 2 (against the original papers)

| # | Config | Finding | Severity |
| --- | --- | --- | --- |
| N1 | `graphtoken-faithful` | Node features are RWSE(16) + LapPE(8) + IDX(16). GraphToken evaluates **LPE, IDX or LPE+IDX, each at d=4** (LPE+IDX d=8, §4.2.2) and **no RWSE**. Not listed in the README deviation row for this arm. | Low |
| N2 | `graphtoken-faithful` | The readout `[mean ‖ sum ‖ h_a ‖ h_b]` is used for every task. The paper picks one readout per task as a hyperparameter: pool (mean **or** sum) for graph tasks, "the representation of each node, optionally concatenated with a graph-level pooling" for node tasks, pool or two node representations for edge tasks. Here it is the union of these, with zero rows when the question names no node. It is a defensible single-model reading and documented. Note: on GraphQA graph-level tasks the node slots are always zero, so the model can tell graph tasks apart from the readout alone. | Low |
| N3 | `graphtoken-faithful` | AdamW at 1e-3 instead of Lion at 0.05, 8 samples per step, GIN only (the paper's GIN was among its weaker encoders, Table 2). All documented. | Low |
| N4 | `node-tokens-faithful` | **The paper and the official code disagree on the hop aggregation.** LLaGA Eq. 3 is the mean over neighbours, `h_v^i = 1/|N_v^1| Σ h_{v'}^{i-1}` (D⁻¹A). `generate_multi_hop_x` @ 288240f uses the symmetric D^-1/2 A D^-1/2. `features.normalized_adjacency` follows the code. This is fine, but the docstring should say "as in the official code; the paper writes the mean". | Low |
| N5 | `node-tokens-faithful` | x_v = mean LLM name embedding ‖ RWSE ‖ LapPE, also propagated. LLaGA HO uses only the text-encoder embedding φ(x_v) (SimTeG by default) and no PE: its Laplacian embedding belongs to the ND template. Documented ("same x_v"). LLaGA trains at lr 2e-5, batch 16, 1 epoch; here the method trains at lr 1e-3, 8 samples per step, with the dataset's epochs. | Low |
| N6 | `-faithful` arms on MetaQA / KQA Pro | **Measured:** the answer string is absent from the question-only prompt in ~100% of cases. Relation labels never reach either encoder: GIN and `Â` are run on the undirected entity graph. GraphToken and LLaGA were never applied to KGQA, so these runs mostly measure closed-book recall + topology. This is documented in the README. But `run_matrix.sh` still runs both arms on `metaqa kqapro` by default: 2 arms × 3 models × 2 datasets of compute for an expected near-floor result. | Medium (compute / interpretation) |
| N7 | `text-colors` | The colour naming is the official `algos.py` mapping, but it is **not monotone in the WL label**: "lime" covers two separate bands (≈0.45–0.5 and ≈0.72–0.78). That contradicts the paper's ordered-colour premise ("variations in hue can be associated with a monotonic structural scale", §4.2). It is faithful to the code, documented in the README, and limited to 6 names (the paper's example uses the same 6). | Low |
| N8 | `text-colors` | README and `text.py` cite "the paper's **Figure 5**" for the prompt and colour names. In arXiv HTML v2 the prompt template is **Figure 3** (Appendix B.3, "Prompt template for max-flow using the CL-OWL method"). Check the figure number against the PDF. | Low |
| N9 | `text-colors` (KQA Pro) | New: the colour block is cut to the first nodes that fit 3,990 prompt tokens (0.6% of prompts). Not in the paper. Documented. | Low |
| N10 | `rel-bias` | Unreachable pairs get their own bucket. GTLM's `SPDBias` indexes `clamp(spd−1, 0, max_spd−1)`, so in GTLM unreachable/far pairs depend on how its preprocessing encodes "no path". Not checked. | Low |

## Per-method detail (pass 2)

### 1. `text`: Less is More
Unchanged code. The attribution is now accurate: the LoRA arm, direct answers, GraphQA adjacency text. The paper's
recipe (instructive CoT, random-word labels, incidence-style serialization) is explicitly not reproduced. **OK.**

### 2. `text-incident`: Talk like a Graph
Unchanged code, still line-for-line with the google-research `incident_encoder` / `create_node_string`.
Neighbour order is documented as unverifiable. **OK.**

### 3. `text-colors`: Colorful Talks (C-OWL)
- **Ordered 1-WL.** Unchanged, and equivalent to `algos.wl` (pass 1).
- **Normalization.** Now `(ℓ − min) / max(1, max − min)`, which is exactly Algorithm 2, lines 3–6.
- **Hue → name.** `cl_owl_color` is `_compute_colors_for_iteration` (hue = 180·x, the simplified HSV→RGB) plus
  `rgb_to_color_name`, copied verbatim and applied to 0–255 RGB. The official repo divides by 255 before its
  0–255 thresholds. Feeding 0–1 values would make everything fall through to the fallback; the reachable set on
  0–255 values is exactly the paper's Figure-3 example colours. This is the right call.
- **Template.**
  - The `<<COLORS>>` explanation is verbatim from the paper's prompt: "shows colors assigned based on WL label
    similarity … Color(i)=c … Colors can be used as a similarity metric among nodes (nodes with similar colors are
    more similar)."
  - The list format is `Color(0):teal, Color(1):lime` with no trailing period, as in the paper.
  - The KG variant substitutes "entity e".
- Remaining notes: N7, N8, N9. C-OWL, not CL-OWL, and LoRA rather than prompting are both documented.
  **OK (faithful to the official code).**

### 4. `prompt-tuning`: P-Tuning
- Code unchanged.
- The citation is now P-Tuning v1 (arXiv 2103.10385), with a note that the paper's encoder is a BiLSTM+MLP while
  this one is peft's MLP.
- The docstring says that this is Less is More's P-Tuning arm, and that GraphToken's soft-prompt baseline
  (Lester et al., frozen LLM, no text) is a different control.

**OK.**

### 5. `graphtoken`, `graphtoken-frozen`, `graphtoken-faithful`: GraphToken
- The **"+ text" arms** are correctly relabelled "GraphToken + text" (Less is More's "Graph Tokens + Text").
  `graphtoken-frozen` no longer claims to be the original setting.
- **`graphtoken-faithful` against the paper:**
  - Frozen LLM: matches §3.2 ("keeping all LLM parameters frozen").
  - Input = graph tokens ‖ question only: matches `Q = E(G) ‖ T(T)` ("eschews representing a graph in text").
    Verified on the data.
  - Learned identity features `nn.Embedding(idx_max, idx_dim)` indexed by node id: match IDX ("unique identity
    encoding designed to break the GNN equivariance", reported as "Learned Features (IDX)"). On GraphQA the node id
    equals the integer name in the question, so the LLM can link "node 3" to its token.
  - Readout includes the queried nodes' representations plus pooling: matches the paper's node- and edge-level
    readouts.
  - A final dense projection to the token space: matches.
  - Remaining notes: N1, N2, N3, and the shared `ToEmbeddingScale`.
- **Grover et al.:** the claim is removed (#10).

**OK for GraphQA. On the KGs see N6.**

### 6. `node-tokens`, `node-tokens-faithful`: LLaGA
- **`node-tokens-faithful` against the paper:**
  - Hop-field overview with `use_hop=4` (5 tokens per node): matches the official HO runs.
  - "Parameters θ of the projector are the only parameters subject to tuning": matches (`regime: frozen`).
  - The projector is the 2-layer MLP, Linear → GELU → Linear (the general-model projector; the classification
    expert used a linear one).
  - No graph text: matches LLaGA's prompt, which carries only the `<graph>` node-embedding sequence.
- Documented adaptations:
  - every node is encoded (LLaGA: the centre node(s));
  - the tokens are prepended instead of filling a placeholder;
  - x_v differs (N5).
- Remaining notes: N4, N5, N6.

**OK for GraphQA.**

### 7. `rel-bias`, `rel-bias-elem`: GTLM SPD bias
- **Same-occurrence zero.** Pairs of tokens of one span occurrence (one node mention, one edge tuple or fact line)
  are forced to 0. This is in the FlexAttention `score_mod` (`occ[q] == occ[k]`) and in the prefill
  `token_bias`.
  - This matches the paper (`b_SPD(u,u) = 0`, Property 2) and the official `SPDBias`. That class is zero-initialized
    and multiplies by `spd > 0`, so distance 0 is never learned.
  - Occurrence ids are unique across packed documents, and padding is −1.
  - "none"-element pairs remain 0.
- **Different occurrences of one element.** Distance 0 between *different* occurrences (e.g. a `Q:` mention and the
  same node in an edge tuple) stays a learned bucket.
  - GTLM has no equivalent: one segment per node, and a prompt node linked by an edge, so distance 1.
  - This is a stated design choice: it is what ties a question mention to the node's facts.
  - In `rel-bias-elem`, this bucket and "other element" are the two learnable buckets.
- **Re-scoping.** The config and docstrings now present `rel-bias` as "GTLM's SPD bias on top of the text model".
  The README and docstring list the GTLM differences explicitly (graph text with sequential RoPE instead of
  per-node RoPE and no edge text; SPD only; Levi graph = GTLM's "incidence" setting). With that framing, the
  remaining differences are by design, not errors.
- Per-layer, per-head tables match the paper's `b_spd ∈ R^{L×H×max_spd}`.
- Remaining note: N10.
- The GPU tests for this path were not re-run in this pass (GPU busy).

**OK, pending the GPU tests.**

### 8. `abs-pe`, `abs-id`
No paper to match. The docstring and config now say that `abs-id` is random, not GraphToken's learned IDX, and point to
`graphtoken-faithful` for that. **OK.**

## Suggested decisions (for you; nothing applied)

1. Run the GPU tests once the queue releases the GPU, in particular `test_rel_bias_same_occurrence_is_zero`,
   `test_rel_bias_train_matches_prefill` and `test_zero_init_is_identity` for the `-faithful` arms.
2. N6: consider `DATASETS=graphqa` for the two `-faithful` arms in `run_matrix.sh`, or keep the KG runs as an
   explicit closed-book floor.
3. N4 and N8: one-line docstring/README corrections (LLaGA: code vs paper normalization; Colorful Talks: figure
   number).
4. N1: add "RWSE + d=8/16 features vs. the paper's LPE/IDX at d=4" to the `graphtoken-faithful` deviation row.
