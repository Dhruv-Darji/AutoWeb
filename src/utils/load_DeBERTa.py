import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class DeBERTaLoader:
    def __init__(
        self, 
        model_name: str = "microsoft/deberta-base",
        model_path: str = None
        ):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        
        print(f"  ⏳ Loading DeBERTa model '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        print(f"  ✅ DeBERTa model '{self.model_name}' loaded successfully.")
    

    def get_embeddings(self, text: str):
        if self.model is None or self.tokenizer is None:
            raise ValueError("DeBERTa model and tokenizer must be loaded before getting embeddings.")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state
        return embeddings
    
    def compute_similarity(self, text1: str, text2: str):
        emb1 = self.get_embeddings(text1).mean(dim=1)
        emb2 = self.get_embeddings(text2).mean(dim=1)
        
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity.item()