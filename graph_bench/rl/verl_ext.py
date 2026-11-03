"""VeRL external_lib for graph GRPO: Jacob's HRM-Text patches, with the thought prefill made explicit.

Load with actor_rollout_ref.model.external_lib=graph_bench.rl.verl_ext (PYTHONPATH must hold this repo and
/work/dfm/marius-ortega/jacob-hrm-text-rlvr). It imports verl_hrm_text.hrm_verl_ext, which applies:
PrefixLM token_type_ids in the FSDP scoring/training passes, FSDP -> vLLM weight fusion (gqkv_proj, gate_up_proj),
the trl and flash_attn compatibility patches, and a thought-channel prefill appended to every prompt.

The prefill is the problem (wiki: mimir-graph-benchmark-plan, RL stack): vLLM's PrefillPrefixLMAttention makes the
whole prefill bidirectional, prefill tokens included, while hrm_verl_ext scores them as causal (and only when its
module global _PREFILL_LEN was set in the same process, which depends on Ray placement). GRAPH_RL_PREFILL picks a
consistent convention instead:
  none    (default) no prefill. Our SFT traces sit in the plain response (the Gemma template strips thought
          channels), so the policy starts reasoning by itself; every completion token is causal in both engines.
  prompt  append "<|channel>thought\\n" and score it as prompt (bidirectional), as vLLM computes it. For RL from a
          model that only reasons when prefilled (arm E3b).
"""
import logging
import os

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

_ORIG_APPLY = SingleTurnAgentLoop.apply_chat_template  # captured before hrm_verl_ext patches it

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

print(f"[graph_bench.rl.verl_ext] thought prefill mode: {MODE} (prefill tokens scored as prompt)", flush=True)
