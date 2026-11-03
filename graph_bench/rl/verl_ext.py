"""VeRL external_lib for graph GRPO: Jacob's HRM-Text patches, with the thought prefill made explicit.

Load with actor_rollout_ref.model.external_lib=graph_bench.rl.verl_ext (PYTHONPATH must hold this repo and
/work/dfm/marius-ortega/jacob-hrm-text-rlvr). It imports verl_hrm_text.hrm_verl_ext, which applies:
PrefixLM token_type_ids in the FSDP scoring/training passes, FSDP -> vLLM weight fusion (gqkv_proj, gate_up_proj),
the trl and flash_attn compatibility patches, and a thought-channel prefill appended to every prompt.

The prefill is the problem (wiki: mimir-graph-benchmark, RL stack): vLLM's PrefillPrefixLMAttention makes the
whole prefill bidirectional, prefill tokens included, while hrm_verl_ext scores them as causal (and only when its
module global _PREFILL_LEN was set in the same process, which depends on Ray placement). GRAPH_RL_PREFILL picks a
consistent convention instead:
  none    (default) no prefill. Our SFT traces sit in the plain response (the Gemma template strips thought
          channels), so the policy starts reasoning by itself; every completion token is causal in both engines.
  prompt  append "<|channel>thought\\n" and score it as prompt (bidirectional), as vLLM computes it. For RL from a
          model that only reasons when prefilled (arm E3b).

GRAPH_RL_FUSE_SYNC: 1 (default) keeps hrm_verl_ext's FSDP -> vLLM weight fusion into gqkv_proj / gate_up_proj, the
names of vLLM's native hrm_text (which also accepts the separate names). 0 sends the separate q/k/v/gate and gate/up
weights, which the Bifrost plugin (/work/dfm/marius-ortega/Bifrost_vllm) requires: it has no fused parameters.
"""
import logging
import os

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

_ORIG_APPLY = SingleTurnAgentLoop.apply_chat_template  # captured before hrm_verl_ext patches it
_ORIG_PER_TENSOR = FSDPEngineWithLMHead.get_per_tensor_param
_ORIG_PREPARE = FSDPEngineWithLMHead.prepare_model_inputs  # verl's own, before hrm_verl_ext's token_type_ids patch

from verl_hrm_text import hrm_verl_ext  # noqa: E402  (applies its patches on import)

_LOG = logging.getLogger(__name__)
THINK_PREFILL = "<|channel>thought\n"
MODE = os.environ.get("GRAPH_RL_PREFILL", "none")
if MODE not in ("none", "prompt"):
    raise ValueError(f"GRAPH_RL_PREFILL={MODE!r}: expected none or prompt")

# hrm_verl_ext's FSDP patch zeroes token_type_ids on the _PREFILL_LEN tokens before the response; with both modes the
# prefill (if any) is prompt, so that length must stay 0. Our appender below never sets it.
hrm_verl_ext._PREFILL_LEN = 0
hrm_verl_ext._PREFILL_IDS = []

if MODE == "none":
    SingleTurnAgentLoop.apply_chat_template = _ORIG_APPLY
else:
    async def _apply_with_prefill(self, messages, **kwargs):
        prompt_ids = await _ORIG_APPLY(self, messages, **kwargs)
        return list(prompt_ids) + list(self.tokenizer.encode(THINK_PREFILL, add_special_tokens=False))

    SingleTurnAgentLoop.apply_chat_template = _apply_with_prefill

def _prepare_with_token_types(self, micro_batch):
    """PrefixLM token_type_ids for the FSDP scoring / training passes: prompt 1 (bidirectional), response 0 (causal).

    Replaces hrm_verl_ext's version, which mis-aligns the response (2026-11-03): verl 0.9 hands a nested response_mask
    holding only each row's response, and hrm_verl_ext pads it on the right to the full row, so the "response" marks
    landed on the first tokens of the prompt and the response stayed bidirectional (rollout-vs-actor prob diff 0.155,
    Pearson 0.77). Here each row's last len(response) valid tokens are marked causal."""
    import torch
    model_inputs, output_args = _ORIG_PREPARE(self, micro_batch)
    am = model_inputs.get("attention_mask")
    if not isinstance(am, torch.Tensor) or am.dim() != 2 or "response_mask" not in micro_batch.keys():
        return model_inputs, output_args  # remove-padding path: no dense layout to mark
    resp = micro_batch["response_mask"]
    lens = ([int(t.numel()) for t in resp.unbind()] if getattr(resp, "is_nested", False)
            else [int(resp.shape[-1])] * am.shape[0])
    tti = am.to(torch.long).clone()
    for row, n in enumerate(lens):
        valid = am[row].nonzero(as_tuple=True)[0]
        if n:
            tti[row, valid[-n:]] = 0
    model_inputs["token_type_ids"] = tti
    return model_inputs, output_args


FSDPEngineWithLMHead.prepare_model_inputs = _prepare_with_token_types

FUSE_SYNC = os.environ.get("GRAPH_RL_FUSE_SYNC", "1")
if FUSE_SYNC not in ("0", "1"):
    raise ValueError(f"GRAPH_RL_FUSE_SYNC={FUSE_SYNC!r}: expected 0 or 1")
if FUSE_SYNC == "0":
    FSDPEngineWithLMHead.get_per_tensor_param = _ORIG_PER_TENSOR
print(f"[graph_bench.rl.verl_ext] weight sync: {'fused (native vLLM hrm_text)' if FUSE_SYNC == '1' else 'separate (Bifrost)'}",
      flush=True)
print(f"[graph_bench.rl.verl_ext] thought prefill mode: {MODE} (prefill tokens scored as prompt)", flush=True)

if os.environ.get("GRAPH_RL_DEBUG_TTI") == "1":
    import torch

    _PATCHED_PREPARE = FSDPEngineWithLMHead.prepare_model_inputs
    _calls = {"n": 0}

    def _runs(row) -> str:
        """Run-length view of a 0/1 row: 1x812 0x256 ..."""
        vals = row.tolist()
        out, cur, n = [], vals[0], 0
        for v in vals:
            if v == cur:
                n += 1
            else:
                out.append(f"{cur}x{n}")
                cur, n = v, 1
        out.append(f"{cur}x{n}")
        return " ".join(out)

    def _debug_prepare(self, micro_batch):
        model_inputs, output_args = _PATCHED_PREPARE(self, micro_batch)
        if _calls["n"] < 2:
            _calls["n"] += 1
            am, tti = model_inputs.get("attention_mask"), model_inputs.get("token_type_ids")
            print(f"[tti-debug] keys={sorted(micro_batch.keys())}", flush=True)
            for key in ("response_mask", "loss_mask", "attention_mask", "input_ids"):
                if key in micro_batch.keys():
                    v = micro_batch[key]
                    print(f"[tti-debug] micro_batch[{key}] nested={getattr(v, 'is_nested', False)} "
                          f"shape={tuple(v.shape) if not getattr(v, 'is_nested', False) else '-'}", flush=True)
            if isinstance(am, torch.Tensor) and isinstance(tti, torch.Tensor):
                print(f"[tti-debug] model attention_mask {tuple(am.shape)} row0: {_runs(am[0].long())}", flush=True)
                print(f"[tti-debug] token_type_ids row0: {_runs(tti[0].long())}", flush=True)
        return model_inputs, output_args

    FSDPEngineWithLMHead.prepare_model_inputs = _debug_prepare
    print("[graph_bench.rl.verl_ext] token_type_ids debug ON", flush=True)
