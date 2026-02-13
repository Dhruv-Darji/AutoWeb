# src/model_interface.py
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Transformers imports
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, logging
logging.set_verbosity_error()

# Optional: if you have bitsandbytes installed and want 8-bit loading,
# keep the code here but it will only run if bitsandbytes is available.
try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False
    
print(f"[VLModel] bitsandbytes available: {BNB_AVAILABLE}")

class VLModel:
    """
    Wrapper for Qwen2-VL-2B-Instruct local inference.

    Usage:
        m = VLModel(model_folder="D:/Environments/Models/Qwen2-VL-2B-Instruct", device="cuda")
        res = m.infer(image_or_pil, prompt_text, max_new_tokens=256)
        print(res["raw_text"])
    """

    def __init__(self,
                 model_folder: str,
                 device: Optional[str] = None,
                 use_8bit: bool = False,
                 dtype: Optional[str] = None):
        """
        Args:
            model_folder: local folder where model and processor are stored
            device: "cuda" or "cpu" or None (auto)
            use_8bit: try to load weights in 8-bit (requires bitsandbytes)
            dtype: optional dtype string: "fp16" or "float32" or None
        """
        # Normalize path to handle Windows paths properly
        self.model_folder = str(Path(model_folder).resolve())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_8bit = use_8bit and BNB_AVAILABLE
        self.dtype = dtype

        # load processor
        print(f"[VLModel] loading processor from {self.model_folder} ...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_folder, 
            trust_remote_code=True,
            local_files_only=True
        )

        # tokenizer (some processors expose tokenizer)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.tokenizer = self.processor.tokenizer
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_folder, 
                    trust_remote_code=True,
                    local_files_only=True
                )
            except Exception:
                self.tokenizer = None

        # model loading strategy: prefer 8-bit if requested & available, else fp16 if GPU else cpu
        model_kwargs: Dict[str, Any] = {}
        if self.use_8bit:
            # requires bitsandbytes & transformers support for load_in_8bit
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            if self.device.startswith("cuda"):
                # try fp16 to save memory
                if self.dtype == "fp16" or self.dtype is None:
                    model_kwargs["torch_dtype"] = torch.float16
                elif self.dtype == "float32":
                    model_kwargs["torch_dtype"] = torch.float32
                model_kwargs["device_map"] = "auto"
            else:
                # CPU: keep default dtype
                model_kwargs["device_map"] = "cpu"

        print(f"[VLModel] loading model with kwargs: {model_kwargs}")
        try:
            # Use AutoModelForVision2Seq for Qwen2-VL (vision-language model with generation)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_folder,
                trust_remote_code=True,
                local_files_only=True,
                **model_kwargs
            )
        except Exception as e:
            print("[VLModel] Warning: model loading with preferred strategy failed:", e)
            print("[VLModel] Falling back to safe CPU load (may be slow).")
            # clear any cached CUDA allocations before fallback
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # fallback: load to CPU
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_folder, 
                trust_remote_code=True,
                local_files_only=True,
                device_map="cpu",
                low_cpu_mem_usage=True
            )

        # ensure model in eval mode
        self.model.eval()

        print(f"[VLModel] model loaded. device = {self.device}, use_8bit={self.use_8bit}")

    # ---------------------
    # Helper: build inputs
    # ---------------------
    def _prepare_inputs(self, image_or_tensor, prompt_text: str, return_tensors: str = "pt"):
        """
        Accepts:
          - image_or_tensor: PIL.Image or torch tensor or already-produced dict of inputs
          - prompt_text: str
        Returns: dict (tensors) ready to pass to model.generate
        """
        # If the caller already passed processor-ready dict (pixel_values etc.), just use it
        if isinstance(image_or_tensor, dict):
            inputs = image_or_tensor
            # make sure text is present (some callers pack text separately)
            if "text" not in inputs and prompt_text is not None:
                inputs["text"] = prompt_text
            return inputs

        # If we got a PIL.Image or numpy array, call processor
        # For Qwen2-VL, processor expects: processor(text=..., images=..., return_tensors="pt")
        inputs = self.processor(text=prompt_text, images=image_or_tensor, return_tensors=return_tensors)

        # --- Sanity check: ensure processor inserted image placeholder tokens when images provided ---
        try:
            input_ids = inputs.get("input_ids", None)
            pixel_values = inputs.get("pixel_values", None)
            # If we have pixel_values (or images) but no image placeholder tokens, try to repair
            if pixel_values is not None and input_ids is not None:
                # convert a few token ids back to strings if tokenizer available
                def _tokens_preview(ids_tensor, tokenizer, n=120):
                    if tokenizer is None:
                        return None
                    ids = ids_tensor[0].tolist()[:n]
                    return tokenizer.convert_ids_to_tokens(ids)

                has_image_tokens = False
                if self.tokenizer is not None:
                    toks = _tokens_preview(input_ids, self.tokenizer, n=512) or []
                    # look for any token that clearly marks an image placeholder
                    for t in toks:
                        if t and ("<image" in t.lower() or "<img" in t.lower() or "[img" in t.lower() or "image" == t.lower()):
                            has_image_tokens = True
                            break

                if not has_image_tokens:
                    # Attempt 1: replace common placeholders with tokenizer-known special token (if any)
                    tried = False
                    if self.tokenizer is not None:
                        candidates = [st for st in getattr(self.tokenizer, "all_special_tokens", []) if "img" in st or "image" in st]
                        if candidates:
                            for cand in candidates:
                                tried = True
                                new_prompt = (prompt_text or "").replace("<image>", cand)
                                repaired = self.processor(text=new_prompt, images=image_or_tensor, return_tensors=return_tensors)
                                rid = repaired.get("input_ids")
                                if rid is not None:
                                    toks2 = _tokens_preview(rid, self.tokenizer, n=512) or []
                                    if any(("<image" in (x or "").lower() or "<img" in (x or "").lower() or "[img" in (x or "").lower()) for x in toks2):
                                        inputs = repaired
                                        has_image_tokens = True
                                        break
                    # Attempt 2: remove the explicit placeholder and let processor inject implicit image tokens
                    if not has_image_tokens and not tried:
                        repaired = self.processor(text=(prompt_text or "").replace("<image>", ""), images=image_or_tensor, return_tensors=return_tensors)
                        rid = repaired.get("input_ids")
                        if rid is not None:
                            if self.tokenizer is not None:
                                toks3 = _tokens_preview(rid, self.tokenizer, n=512) or []
                                if any(("<image" in (x or "").lower() or "<img" in (x or "").lower() or "[img" in (x or "").lower()) for x in toks3):
                                    inputs = repaired
                                    has_image_tokens = True

                    # If still no image tokens but pixel_values present, raise a clearer error (with diagnostics)
                    if not has_image_tokens:
                        preview = None
                        if self.tokenizer is not None and input_ids is not None:
                            preview = _tokens_preview(input_ids, self.tokenizer, n=120)
                        raise ValueError(
                            "Processor returned image features but the tokenized prompt contains NO image placeholder tokens.\n"
                            f"This typically means the prompt's image placeholder (e.g. '<image>') is not recognized by the model tokenizer.\n\n"
                            "Diagnostics:\n"
                            f"  - prompt_text (truncated): {str(prompt_text)[:200]!r}\n"
                            f"  - input_ids length: {input_ids.shape if hasattr(input_ids, 'shape') else 'n/a'}\n"
                            f"  - pixel_values shape: {pixel_values.shape if hasattr(pixel_values, 'shape') else 'n/a'}\n"
                            f"  - token preview (first 120 tokens): {preview}\n\n"
                            "Quick fixes: ensure the prompt uses the model's exact image placeholder (commonly '<image>'),\n"
                            "or try running the script with --demo after updating the model's processor/tokenizer."
                        )
        except Exception:
            # Don't crash preprocessing for non-fatal introspection errors; let generation surface the issue
            pass

        # Move inputs to model device
        try:
            # Get the device where model parameters are
            device = next(self.model.parameters()).device
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)
        except Exception:
            # If device detection fails, let device_map handle it
            pass

        return inputs

    def get_preferred_image_key(self, sample_image=None) -> str:
        """Return the best image placeholder to use in prompts for this model.

        Tries (in order):
        - tokenizer special tokens that contain 'img'/'image'
        - the literal '<image>' if tokenizer recognizes it
        - let the processor inject implicit image tokens (returns empty string)

        The returned string should be passed to `PromptEngine.build_prompt(image_key=...)`.
        Returns empty string when no explicit placeholder is required/recognized.
        """
        # 1) Look for obvious special tokens from tokenizer
        try:
            if getattr(self, "tokenizer", None) is not None:
                for st in getattr(self.tokenizer, "all_special_tokens", []) or []:
                    if "img" in st.lower() or "image" in st.lower():
                        return st

                # 2) check whether the literal '<image>' maps to token ids
                try:
                    ids = self.tokenizer.convert_tokens_to_ids(["<image>"])
                    if ids and ids[0] != self.tokenizer.unk_token_id:
                        return "<image>"
                except Exception:
                    # tokenizer may not recognize the string-to-id mapping; ignore
                    pass

        except Exception:
            # non-fatal
            pass

        # 3) As a final check, ask the processor to tokenize a small prompt and inspect whether
        # it inserted any image-like tokens. If so, return '<image>' as a hint; otherwise
        # return empty string to indicate the processor should handle images implicitly.
        try:
            probe_prompt = "<image>"
            proc_out = None
            if sample_image is not None:
                proc_out = self.processor(text=probe_prompt, images=sample_image, return_tensors="pt")
            else:
                # If no sample image, call processor with an empty PIL image of expected size
                from PIL import Image
                img = Image.new("RGB", (32, 32), color=(128, 128, 128))
                proc_out = self.processor(text=probe_prompt, images=img, return_tensors="pt")

            input_ids = proc_out.get("input_ids", None)
            if input_ids is not None and getattr(self, "tokenizer", None) is not None:
                toks = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[:256])
                for t in toks:
                    if t and ("<image" in t.lower() or "<img" in t.lower() or "[img" in t.lower() or "image" == t.lower()):
                        return "<image>"
        except Exception:
            pass

        # No explicit placeholder detected — prefer implicit handling
        return ""

    # ---------------------
    # Infer / generate
    # ---------------------
    def infer(self,
              image_or_tensor: Union[Any, Dict],
              prompt_text: str,
              max_new_tokens: int = 128,
              do_sample: bool = False,
              top_p: float = 0.95,
              temperature: float = 0.2,
              return_tensors: str = "pt") -> Dict[str, Any]:
        """
        Run inference and return decoded text + latency.

        Args:
            image_or_tensor: PIL.Image (preferred) or dict of precomputed tensors
            prompt_text: prompt string (contains image placeholder)
            max_new_tokens: tokens to generate
            do_sample: whether to sample
            top_p, temperature: sampling params (if sampling)
        Returns:
            {"output_text": str, "raw_text": str, "latency": float, "generated_ids": tensor (optional)}
        """
        start = time.time()
        inputs = self._prepare_inputs(image_or_tensor, prompt_text, return_tensors=return_tensors)

        # Generation kwargs
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            return_dict_in_generate=True
        )
        # add eos_token_id only when tokenizer exposes it
        try:
            if getattr(self, 'tokenizer', None) is not None and getattr(self.tokenizer, 'eos_token_id', None) is not None:
                generate_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        except Exception:
            pass

        # --- Diagnostics: print input tensor shapes and token preview before generation ---
        try:
            model_dev = next(self.model.parameters()).device
        except Exception:
            model_dev = 'unknown'
        try:
            input_ids = inputs.get('input_ids', None)
            pixel_values = inputs.get('pixel_values', None)
            print(f"[VLModel] generate() model_device={model_dev}, input_ids={getattr(input_ids, 'shape', None)}, pixel_values={getattr(pixel_values, 'shape', None)}")
            if input_ids is not None and getattr(self, 'tokenizer', None) is not None:
                toks_preview = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[:120])
                # print(f"[VLModel] token preview (first 120 tokens): {toks_preview}")
                if len(input_ids[0]) > 4096:
                    print('[VLModel] ⚠ input token length > 4096 — this may cause very long generation or memory pressure')
        except Exception as _:
            pass

        # print(f"[VLModel] generate kwargs: {generate_kwargs}")

        # Run generation (catch common image/token mismatches and provide actionable message)
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
        except ValueError as e:
            msg = str(e)
            if "Image features and image tokens do not match" in msg or "image features and image tokens" in msg.lower():
                # Improve the error with specific guidance
                raise ValueError(
                    "Model generation failed because image features and token placeholders are misaligned.\n"
                    "Likely causes: the prompt's image placeholder (e.g. '<image>') is not the exact special token the model tokenizer expects,\n"
                    "or the processor/tokenizer pair from the model checkpoint is incompatible with the prompt format.\n\n"
                    "Immediate remedies:\n"
                    "  1) Ensure your prompt contains the model's image placeholder token (common: '<image>') exactly.\n"
                    "  2) Try removing the explicit '<image>' from the prompt so the processor can insert implicit image tokens.\n"
                    "  3) If you use a custom prompt, inspect tokenizer.special_tokens or run with VLMODEL_DEBUG=1 to print token diagnostics.\n\n"
                    f"Original error: {msg}"
                ) from e
            raise
        except (RuntimeError, torch.cuda.OutOfMemoryError) as oom_err:
            # Attempt graceful recovery for OOM during generation: free cache and retry once with reduced token budget
            try:
                print("[VLModel] CUDA out of memory during generate(): freeing cache and retrying with reduced max_new_tokens...")
                torch.cuda.empty_cache()
            except Exception:
                pass

            # reduce token budget and retry once
            if generate_kwargs.get("max_new_tokens", 0) > 16:
                reduced = max(16, generate_kwargs.get("max_new_tokens") // 2)
                generate_kwargs["max_new_tokens"] = reduced
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **generate_kwargs)
                except Exception as retry_exc:
                    # still failed — raise a clearer error
                    raise RuntimeError(
                        "Generation failed due to CUDA OOM even after retry with reduced max_new_tokens. "
                        "Consider lowering batch size/max_new_tokens, enabling 8-bit (USE_8BIT), or switching to CPU/quantized model."
                    ) from retry_exc
            else:
                raise RuntimeError(
                    "Generation failed due to CUDA OOM. Consider lowering max_new_tokens, enabling 8-bit (USE_8BIT), "
                    "or switching to CPU/quantized model."
                ) from oom_err
        
        # Decode output — handle both ReturnDict (sequences) and raw tensor returns
        try:
            seq_tensor = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0]
        except Exception:
            seq_tensor = outputs[0]

        try:
            # Try processor.decode first (preferred for vision models)
            raw_text = self.processor.decode(seq_tensor, skip_special_tokens=True)
        except Exception:
            # Fallback to tokenizer if processor doesn't have decode
            if self.tokenizer is not None:
                raw_text = self.tokenizer.decode(seq_tensor, skip_special_tokens=True)
            else:
                raw_text = str(seq_tensor.tolist())

        try:
            input_len = inputs.get("input_ids", None).shape[1] if inputs.get("input_ids", None) is not None else 0
            generated_tokens = seq_tensor[input_len:] if input_len < seq_tensor.shape[0] else seq_tensor
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except Exception:
            if self.tokenizer is not None:
                output_text = self.tokenizer.decode(seq_tensor, skip_special_tokens=True)
            else:
                output_text = str(seq_tensor.tolist())

        latency = time.time() - start
        return {
            "output_text": output_text, 
            "raw_text": raw_text, 
            "latency": latency, 
            "generated_ids": outputs
            }

