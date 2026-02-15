import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class DeBERTaLoader:
    """Cross-encoder wrapper around DeBERTa.

    Instead of producing independent sentence embeddings (bi-encoder),
    this feeds *both* texts as a pair
        [CLS] text_a [SEP] text_b [SEP]
    and derives a relevance score from the CLS representation.
    This mirrors the SeeAct cross-encoder ranking approach.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-base",
        model_path: str = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        # Learnable projection head (initialised after model is loaded)
        self._score_head = None

    def load_model(self):
        print(f"  ⏳ Loading DeBERTa model '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # Force CPU — DeBERTa-base is small (~440 MB) and fast on CPU.
        # Keeping it off the GPU leaves VRAM free for the Qwen VL model.
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to("cpu")
        self.model.eval()

        # A simple linear head that maps the CLS hidden-state → scalar score.
        # Weights are randomly initialised (Xavier) which is fine for
        # zero-shot ranking: CLS already encodes cross-attention between
        # the two segments, so even without task-specific fine-tuning it
        # produces much better ranking than bi-encoder cosine.
        hidden_size = self.model.config.hidden_size  # 768 for deberta-base
        self._score_head = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self._score_head.weight)
        torch.nn.init.zeros_(self._score_head.bias)
        self._score_head.to("cpu")
        self._score_head.eval()

        print(f"  ✅ DeBERTa model '{self.model_name}' loaded successfully (cross-encoder mode, CPU).")

    # ------------------------------------------------------------------
    # Primary API – cross-encoder relevance score
    # ------------------------------------------------------------------
    def compute_cross_score(self, plan: str, element_repr: str) -> float:
        """Score a (plan, element) pair using cross-encoder attention.

        The two texts are fed as a single sequence separated by [SEP].
        The CLS token's hidden state — which attends to both segments —
        is projected to a scalar relevance score.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "DeBERTa model and tokenizer must be loaded before computing scores."
            )

        inputs = self.tokenizer(
            plan,
            element_repr,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        # Ensure inputs are on CPU (matches model placement)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden]
            score = self._score_head(cls_embedding).squeeze(-1)  # scalar

        return score.item()

    def compute_cross_scores_batch(
        self, plan: str, element_reprs: list[str], batch_size: int = 64
    ) -> list[float]:
        """Score many (plan, element) pairs efficiently in batches."""
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "DeBERTa model and tokenizer must be loaded before computing scores."
            )

        all_scores: list[float] = []
        for start in range(0, len(element_reprs), batch_size):
            batch = element_reprs[start : start + batch_size]
            inputs = self.tokenizer(
                [plan] * len(batch),
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            # Ensure inputs are on CPU (matches model placement)
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
                scores = self._score_head(cls_embeddings).squeeze(-1)  # [B]
            all_scores.extend(scores.tolist() if scores.dim() > 0 else [scores.item()])
        return all_scores

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward-compatibility)
    # ------------------------------------------------------------------
    def get_embeddings(self, text: str):
        """Return raw last-hidden-state embeddings (bi-encoder style).

        ⚠️  Deprecated – prefer `compute_cross_score` for ranking.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "DeBERTa model and tokenizer must be loaded before getting embeddings."
            )

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state

    def compute_similarity(self, text1: str, text2: str):
        """⚠️  Deprecated – bi-encoder cosine similarity. Use compute_cross_score."""
        emb1 = self.get_embeddings(text1).mean(dim=1)
        emb2 = self.get_embeddings(text2).mean(dim=1)

        similarity = F.cosine_similarity(emb1, emb2)
        return similarity.item()