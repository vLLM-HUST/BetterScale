# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate only: native DSV4 forward with two residual clones removed.

Adapted from pinned vllm-ascend 9bf964c models/deepseek_v4.py. HC-pre only reads
its input and HC-post allocates a new output; neither residual is an in-place
workspace. All actual attention, norm and MLP implementations remain native.
Install before compilation. This is not enabled by the public Worker until the
whole-forward alias and graph-pool benefit gates pass.
"""


def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
    residual = hidden_states
    hidden_states, post, comb = self.hc_pre(
        hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
    )
    hidden_states = self.input_layernorm(hidden_states)
    attn_kwargs = {
        "positions": positions,
        "hidden_states": hidden_states,
        "llama_4_scaling": llama_4_scaling,
    }
    hidden_states = self.self_attn(**attn_kwargs)
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    residual = hidden_states
    hidden_states, post, comb = self.hc_pre(
        hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    return hidden_states, residual


def install():
    # Native keeps the legacy V2 class name inside its deepseek_v4 module.
    from vllm_ascend.models.deepseek_v4 import DeepseekV2DecoderLayer

    DeepseekV2DecoderLayer.forward = forward
