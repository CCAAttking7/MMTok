# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
MMTok for Qwen2.5-VL: token selection injection and patching.
"""

from .qwen2_5_vl_mmtok import mmtok_qwen2_5_vl

__all__ = ["mmtok_qwen2_5_vl"]
