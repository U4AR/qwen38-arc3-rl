"""Idempotent patch for the pinned vLLM 0.17.2rc1 (applied by serve.sh on every start).

Default max_tokens ("rest of the context", as the Duck uses) let sequences run
to exactly max_model_len; with MTP speculative decoding drafting past the end,
the engine died with a device-side assert in flash-attention. Stop
CONTEXT_EDGE_MARGIN tokens short of the limit instead.
"""
import sys
from pathlib import Path

import vllm.entrypoints.utils as mod

path = Path(mod.__file__)
src = path.read_text()
old = "    model_max_tokens = max_model_len - input_length\n"
new = (
    "    # patched (qwen38-arc3-rl): keep speculative drafts inside the context window\n"
    "    import os as _os\n"
    "    _margin = int(_os.environ.get('CONTEXT_EDGE_MARGIN', '16'))\n"
    "    if max_model_len - input_length <= _margin:\n"
    "        raise ValueError(\n"
    "            f\"Input length ({input_length}) exceeds model's maximum \"\n"
    "            f\"context length ({max_model_len - _margin} usable).\"\n"
    "        )\n"
    "    model_max_tokens = max_model_len - input_length - _margin\n"
)
if "patched (qwen38-arc3-rl)" in src:
    print("vllm already patched")
elif old in src:
    path.write_text(src.replace(old, new, 1))
    print("vllm patched:", path)
else:
    sys.exit("vllm patch target not found; check vLLM version")
