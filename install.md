# MMTok Installation (dependency order)

## 1. Create environment

```bash
conda create -n mmtok python=3.12 -y
conda activate mmtok
```

## 2. Install PyTorch and base deps

```bash
# PyTorch (tested with 2.8.0 + CUDA 12.8 on H100 and A6000; newer versions may cause issues)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## 3. Install LLaVA-NeXT (if using LLaVA; use a directory of your choice)

> **Note:** Install LLaVA-NeXT only (not the older haotian-liu/LLaVA).
> LLaVA-NeXT can load both LLaVA-1.5 and LLaVA-NeXT model weights.
> Use `transformers==4.52.4`; newer versions may cause errors.

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]" --no-deps
cd ..
```



## 4. Install MMTok (from this repo root)

```bash
pip install -e .
```

Verify installation (should print `mmtok installed successful`):

```bash
python -c "import mmtok; print('mmtok installed successful')"
```

## 5. Run example

```bash
# For LlaVA-NeXT-7B
python example/llava_mmtok_example.py

# For Qwen-2.5-VL-7B
python example/qwen_mmtok_example.py
```

## 6. Optional: lmms-eval (for evaluation)

Install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) following the [official installation guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main):

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e . --no-deps
```

To use MMTok with `lmms-eval`, you simply need to **wrap the model instance** in its corresponding evaluation file within `lmms-eval/lmms_eval/models/`. Reference implementations: **[example/lmms_eval_llava_mmtok.py](example/lmms_eval_llava_mmtok.py)** (LLaVA) and **[example/lmms_eval_qwen_mmtok.py](example/lmms_eval_qwen_mmtok.py)** (Qwen2.5-VL).

#### Step 1: Add the Import

Add the MMTok integration at the top of your model file (e.g., `llava.py` or `qwen2_5_vl.py`):

```python
from mmtok import mmtok
# For Qwen2.5-VL specifically, use the specialized wrapper
from mmtok.qwen.qwen2_5_vl_mmtok import mmtok_qwen2_5_vl
```

#### Step 2: Wrap the Model

Locate the section where the model and tokenizer are initialized (typically in the `__init__` or `_create_model` method) and apply the wrapper.

| Model | Implementation | Key Parameter |
| --- | --- | --- |
| **LLaVA** | `self._model = mmtok(self._model, ...)` | `target_vision_tokens=32` (absolute count) |
| **Qwen2.5-VL** | `self._model, self.processor = mmtok_qwen2_5_vl(...)` | `retain_ratio=0.1` (fraction, e.g., 10%) |

#### Step 3: Code Snippets

**For LLaVA (`llava.py`):**

```python
# MMTok: Ensure fast tokenizer for coverage computation, then wrap
# ⚠️ Important: Fast tokenizer is required for MMTok's coverage computation
if not getattr(self._tokenizer, 'is_fast', False):
    from transformers import AutoTokenizer
    self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)

self._model = mmtok(self._model, language_tokenizer=self._tokenizer, target_vision_tokens=32)
```

**For Qwen2.5-VL (`qwen2_5_vl.py`):**

```python
# MMTok: wrap both model and processor

self._model, self.processor = mmtok_qwen2_5_vl(
    self._model, 
    language_tokenizer=self._tokenizer, 
    processor=self.processor, 
    retain_ratio=0.1
)
```

## 7. Optional: Flash Attention

We tested that with the versions above (e.g. torch 2.8.0), Flash Attention 2 can be installed relatively easily. With newer torch versions, installing flash-attn may run into environment or build issues that you may need to resolve.

```bash
pip install flash-attn --no-build-isolation
```

Then in `example/llava_mmtok_example.py`, change `attn_implementation` to:

```python
attn_implementation="flash_attention_2"
```

## 8. Optional: Qwen-VL

```bash
pip install qwen-vl-utils
```

If you see `get_model_name_from_path` is not defined, try `pip install transformers>=4.52.4`.

