#  Main class to load the Dataset
import os
import pandas as pd
from PIL import Image
from io import BytesIO

class Mind2WebDataset:
    
    """
    Loader for Multimodal-Mind2Web dataset stored in parquet + image-bytes format.
    Supports sampling a subset of rows for efficient local experiments.
    """
    
    def __init__(self, root_dir: str, split: str = "train", sample_frac: float = None, sample_size: int = None, shuffle: bool = True, seed: int = 42):
        """
        Args:
            root_dir: root folder where dataset is downloaded
            split: which split to load: "train", "test_task", "test_website", ...
            sample_frac: if given, fraction of dataset to sample (e.g. 0.1 for 10%)
            sample_size: if given, exact number of samples to randomly sample
            shuffle: whether to shuffle before sampling
            seed: random seed
        """
        self.root = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, "data")

        self.parquet_files = [
            os.path.join(self.split_dir, fname)
            for fname in os.listdir(self.split_dir)
            if fname.startswith(split) and fname.endswith(".parquet")
        ]

        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found for split {split} in {self.split_dir}")
        
        dfs = []

        for f in self.parquet_files:
            df = pd.read_parquet(f)
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        print(f"[MultiModal-Mind2Web] Loaded split '{split}' with {len(self.df)} rows (raw).")

        print(f"[MultiModal-Mind2Web] Columns: {self.df.columns.tolist()}")

        if sample_frac is not None:
            self.df = self.df.sample(frac=sample_frac, random_state=seed)
        
        elif sample_size is not None:
            self.df = self.df.sample(n=sample_size, random_state=seed)

        if shuffle:
            self.df = self.df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def _load_image(self, screenshot_field):
        """
        screenshot_field: dict, expected keys 'bytes' or 'path'
        returns: PIL.Image
        """
        if isinstance(screenshot_field, dict):
            b = screenshot_field.get("bytes", None)
            if b:
                img = Image.open(BytesIO(b))                
                return img.convert("RGB")
            else:
                p = screenshot_field.get("path", None)
                if p is None:
                    raise ValueError("Screenshot dict has no bytes or path.")
                abs_path = os.path.join(self.root, p)
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"Image path not found: {abs_path}")
                img = Image.open(abs_path)
                return img.convert("RGB")
        elif isinstance(screenshot_field, str):
            abs_path = os.path.join(self.root, screenshot_field)
            img = Image.open(abs_path)
            return img.convert("RGB")
        else:
            raise ValueError("Unexpected format for screenshot field: %s" % type(screenshot_field))


    def __getitem__(self, idx: int):
        """
        Returns: dics with keys:
            'image' : PIL.Image
            'raw_html' : str
            'operation' : dict
            'pos_candidates' : list of dicts
            'task' : str
            'metadata' : dict (website, annotation_id etc.)
        """

        row = self.df.iloc[idx]

        sample = {}
        sample["image"] = self._load_image(row["screenshot"])
        sample["raw_html"] = row.get("raw_html", None)
        sample["cleaned_html"] = row.get("cleaned_html", None)
        sample["operation"] = row["operation"]  # dict
        sample["candidates"] = row.get("pos_candidates", [])
        sample["task"] = row.get("confirmed_task", "")
        sample["metadata"] = {
            "action_uid": row.get("action_uid"),
            "annotation_id": row.get("annotation_id"),
            "website": row.get("website"),
            "domain": row.get("domain"),
        }

        return sample