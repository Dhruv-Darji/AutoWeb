# src/model_interface.py
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

# Transformers imports
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
            # fallback: load to CPU
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_folder, 
                trust_remote_code=True,
                local_files_only=True,
                device_map="cpu"
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

    # ---------------------
    # Infer / generate
    # ---------------------
    def infer(self,
              image_or_tensor: Union[Any, Dict],
              prompt_text: str,
              max_new_tokens: int = 256,
              do_sample: bool = False,
              top_p: float = 0.95,
              temperature: float = 0.0,
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
            {"raw_text": str, "latency": float, "generated_ids": tensor (optional)}
        """
        start = time.time()
        inputs = self._prepare_inputs(image_or_tensor, prompt_text, return_tensors=return_tensors)

        # Generation kwargs
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,  # temperature must be > 0
            top_p=top_p
        )

        # Run generation
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        # Decode output
        try:
            # Try processor.decode first (preferred for vision models)
            raw_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception:
            # Fallback to tokenizer if processor doesn't have decode
            if self.tokenizer is not None:
                raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                raw_text = str(outputs[0].tolist())

        latency = time.time() - start
        return {"raw_text": raw_text, "latency": latency, "generated_ids": outputs}

