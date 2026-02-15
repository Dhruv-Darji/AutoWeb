import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Pre-trained cross-encoder for semantic ranking (much stronger than random head)
try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


def _pick_device() -> str:
    """Choose best available device for the cross-encoder (~80 MB, fits easily on any GPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DeBERTaLoader:
    """Cross-encoder wrapper for DOM-element ranking.

    Two modes:
      • **Pre-trained cross-encoder** (default, recommended):
        Uses ``sentence-transformers`` ``CrossEncoder`` class with a model
        trained for semantic ranking (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
        This produces real relevance scores out-of-the-box — no fine-tuning needed.
        Loaded from a **local folder** pointed to by ``CROSSENCODER_MODEL_PATH``
        in ``.env`` (e.g. ``D:/Environments/Models/cross-encoder-MiniLM-L6-v2``).
        Runs on **CUDA** when available (~80 MB VRAM).

      • **Legacy DeBERTa-base** (``use_pretrained_crossencoder=False``):
        Loads ``AutoModel`` + random linear head. Kept for backward-compat.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-base",
        model_path: str = None,
        use_pretrained_crossencoder: bool = True,
        crossencoder_name: str = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.use_pretrained_crossencoder = use_pretrained_crossencoder and _ST_AVAILABLE

        # Resolve cross-encoder model path: .env > arg > HuggingFace hub name
        if crossencoder_name is None:
            from AutoWeb.src.config import get_crossencoder_model_path
            self.crossencoder_name = get_crossencoder_model_path()
        else:
            self.crossencoder_name = crossencoder_name

        self.device = _pick_device()

        # Legacy DeBERTa fields
        self.model = None
        self.tokenizer = None
        self._score_head = None

        # sentence-transformers CrossEncoder
        self._cross_encoder: "STCrossEncoder | None" = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self):
        if self.use_pretrained_crossencoder:
            self._load_pretrained_crossencoder()
        else:
            self._load_legacy_deberta()

    def _load_pretrained_crossencoder(self):
        """Load a purpose-trained cross-encoder from local path onto best device."""
        from AutoWeb.src.logger import logger
        logger.info(f"  ⏳ Loading pre-trained cross-encoder from '{self.crossencoder_name}' → {self.device}...")
        self._cross_encoder = STCrossEncoder(self.crossencoder_name, device=self.device)
        logger.info(f"  ✅ Cross-encoder loaded on {self.device} (~80 MB VRAM).")

    def _load_legacy_deberta(self):
        """Load DeBERTa-base with a random projection head (legacy path)."""
        from AutoWeb.src.logger import logger
        logger.info(f"  ⏳ Loading DeBERTa model '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to("cpu")
        self.model.eval()

        hidden_size = self.model.config.hidden_size
        self._score_head = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self._score_head.weight)
        torch.nn.init.zeros_(self._score_head.bias)
        self._score_head.to("cpu")
        self._score_head.eval()
        logger.info(f"  ✅ DeBERTa model '{self.model_name}' loaded (legacy cross-encoder, CPU).")

    # ------------------------------------------------------------------
    # Primary API — cross-encoder relevance score
    # ------------------------------------------------------------------
    def compute_cross_score(self, plan: str, element_repr: str) -> float:
        """Score a (plan, element) pair."""
        if self.use_pretrained_crossencoder and self._cross_encoder is not None:
            scores = self._cross_encoder.predict([(plan, element_repr)])
            return float(scores[0])
        return self._legacy_cross_score(plan, element_repr)

    def compute_cross_scores_batch(
        self, plan: str, element_reprs: list[str], batch_size: int = 128
    ) -> list[float]:
        """Score many (plan, element) pairs efficiently."""
        if self.use_pretrained_crossencoder and self._cross_encoder is not None:
            pairs = [(plan, er) for er in element_reprs]
            scores = self._cross_encoder.predict(pairs, batch_size=batch_size)
            return [float(s) for s in scores]
        return self._legacy_cross_scores_batch(plan, element_reprs, batch_size)

    # ------------------------------------------------------------------
    # Legacy DeBERTa scoring (kept for backward-compat)
    # ------------------------------------------------------------------
    def _legacy_cross_score(self, plan: str, element_repr: str) -> float:
        if self.model is None or self.tokenizer is None:
            raise ValueError("DeBERTa model must be loaded first.")

        inputs = self.tokenizer(
            plan, element_repr,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            score = self._score_head(cls).squeeze(-1)
        return score.item()

    def _legacy_cross_scores_batch(
        self, plan: str, element_reprs: list[str], batch_size: int = 64
    ) -> list[float]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("DeBERTa model must be loaded first.")

        all_scores: list[float] = []
        for start in range(0, len(element_reprs), batch_size):
            batch = element_reprs[start : start + batch_size]
            inputs = self.tokenizer(
                [plan] * len(batch), batch,
                return_tensors="pt", truncation=True, max_length=512, padding=True,
            )
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls = outputs.last_hidden_state[:, 0, :]
                scores = self._score_head(cls).squeeze(-1)
            all_scores.extend(scores.tolist() if scores.dim() > 0 else [scores.item()])
        return all_scores

    # ------------------------------------------------------------------
    # Deprecated helpers
    # ------------------------------------------------------------------
    def get_embeddings(self, text: str):
        """⚠️ Deprecated — use compute_cross_score."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("DeBERTa model must be loaded first.")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def compute_similarity(self, text1: str, text2: str):
        """⚠️ Deprecated — use compute_cross_score."""
        emb1 = self.get_embeddings(text1).mean(dim=1)
        emb2 = self.get_embeddings(text2).mean(dim=1)
        return F.cosine_similarity(emb1, emb2).item()