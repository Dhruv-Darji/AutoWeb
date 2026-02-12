#  Main class to load the Dataset
import os
import pandas as pd
from PIL import Image
from io import BytesIO

class Mind2WebDataset:
    
    """
    Loader for Multimodal-Mind2Web dataset stored in parquet + image-bytes format.
    Supports sampling a subset of rows for efficient local experiments.

    DESCRIBE data

    | #  | column                | type                                        | nullable | description |
    |----|-----------------------|---------------------------------------------|----------|-------------|
    | 1  | action_uid            | VARCHAR                                     | YES      | Unique action/record identifier
    | 2  | raw_html              | VARCHAR                                     | YES      | Original page HTML
    | 3  | cleaned_html          | VARCHAR                                     | YES      | Preprocessed / cleaned HTML
    | 4  | operation             | VARCHAR                                     | YES      | Raw operation/annotation (dict/JSON)
    | 5  | pos_candidates        | VARCHAR[]                                   | YES      | Positive candidate actions/selectors
    | 6  | neg_candidates        | VARCHAR[]                                   | YES      | Negative candidate actions
    | 7  | website               | VARCHAR                                     | YES      | Website hostname or URL
    | 8  | domain                | VARCHAR                                     | YES      | Domain / site category
    | 9  | subdomain             | VARCHAR                                     | YES      | Subdomain (if available)
    | 10 | annotation_id         | VARCHAR                                     | YES      | Annotation / task identifier (groups rows)
    | 11 | confirmed_task        | VARCHAR                                     | YES      | Human-confirmed task/instruction
    | 12 | screenshot            | STRUCT(bytes BLOB, path VARCHAR)            | YES      | Screenshot image bytes or filesystem path
    | 13 | action_reprs          | VARCHAR[]                                   | YES      | Action representations (stringified)
    | 14 | target_action_index   | VARCHAR                                     | YES      | Index of the correct/target action
    | 15 | target_action_reprs   | VARCHAR                                     | YES      | Representation of the target action
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


    def _extract_oracle_action(self, operation: dict) -> dict:
        """
        Extract oracle (ground-truth) action from operation dict.
        
        Returns standardized action dict:
        {
            'action_type': str (click/type/scroll/select/...),
            'target': {
                'selector': str,
                'text': str,
                'bbox': [x, y, w, h] or None
            },
            'value': str or None
        }
        """
        if not operation or not isinstance(operation, dict):
            return {
                "action_type": "noop",
                "target": None,
                "value": None
            }
        
        # Extract action type
        action_type = operation.get("op", "")
        
        # Map operation types to standard action types
        type_mapping = {
            "CLICK": "click",
            "TYPE": "type",
            "SELECT": "click",  # SELECT is treated as click in SeeAct
            "SCROLL": "scroll",
            "HOVER": "click",  # HOVER treated as click
        }
        
        action_type_std = type_mapping.get(action_type.upper(), action_type.lower())
        
        # Extract target information
        target = None
        if "original_op_before_modification" in operation:
            element = operation["original_op_before_modification"]
        else:
            element = operation
        
        # Try to extract selector and other target info
        selector = element.get("selector") or element.get("element_css_selector")
        text = element.get("text") or element.get("element_text", "")
        
        if selector or text:
            target = {
                "selector": selector,
                "text": text
            }
        
        # Extract value for type operations
        value = operation.get("value") or operation.get("action_input")
        
        return {
            "action_type": action_type_std,
            "target": target,
            "value": value
        }

    def __getitem__(self, idx: int):
        """
        Returns: dics with keys:
            'image' : PIL.Image
            'instruction' : str (task description)
            'oracle_action' : dict (ground-truth action)
            'raw_html' : str
            'cleaned_html' : str
            'operation' : dict (raw operation)
            'candidates' : list of dicts
            'metadata' : dict (website, annotation_id etc.)
        """

        row = self.df.iloc[idx]

        sample = {}
        sample["image"] = self._load_image(row["screenshot"])
        sample["instruction"] = row.get("confirmed_task", "")
        sample["raw_html"] = row.get("raw_html", None)
        sample["cleaned_html"] = row.get("cleaned_html", None)
        sample["operation"] = row["operation"]  # dict (raw)
        sample["candidates"] = row.get("pos_candidates", [])
        
        # Extract oracle action for evaluation
        sample["oracle_action"] = self._extract_oracle_action(row["operation"])
        
        sample["metadata"] = {
            "action_uid": row.get("action_uid"),
            "annotation_id": row.get("annotation_id"),
            "website": row.get("website"),
            "domain": row.get("domain"),
        }

        return sample
    
    def __getTask__(self, idx: int):
        """
        Given a row index `idx`, return the entire task (all rows sharing the same
        `annotation_id`) as a list of samples (same structure as `__getitem__`).
        """
        row = self.df.iloc[idx]
        annotation_id = row.get("annotation_id")
        if annotation_id is None:
            raise ValueError(f"Row at index {idx} has no 'annotation_id'.")
        return self.get_task(annotation_id)

    def get_task(self, annotation_id: str, file_name: str = None) -> list:
        """
        Return all samples that belong to the same task (rows with the given
        `annotation_id`).

        Args:
            annotation_id: annotation id to look up
            file_name: optional parquet filename (basename or full path). If
                       provided, search only that parquet file instead of the
                       concatenated `self.df`.

        Returns:
            List of sample dicts (each identical in structure to `__getitem__` output).
        """
        # Choose search DataFrame: either full concatenated df or specific file
        if file_name:
            matched_paths = [p for p in self.parquet_files if os.path.basename(p) == file_name or p.endswith(file_name)]
            if not matched_paths:
                raise FileNotFoundError(f"No parquet file matching '{file_name}' in {self.split_dir}")
            dfs = [pd.read_parquet(p) for p in matched_paths]
            search_df = pd.concat(dfs, ignore_index=True)
        else:
            search_df = self.df

        matches = search_df[search_df.get("annotation_id") == annotation_id]
        if matches.empty:
            raise KeyError(f"No rows found with annotation_id='{annotation_id}'")

        results = []
        for _, r in matches.iterrows():
            sample = {}
            sample["annotation_id"] = r.get("annotation_id")
            sample["action_uid"] = r.get("action_uid")
            sample["image"] = self._load_image(r["screenshot"])
            sample["instruction"] = r.get("confirmed_task", "")
            sample["raw_html"] = r.get("raw_html", None)
            sample["cleaned_html"] = r.get("cleaned_html", None)
            sample["operation"] = r["operation"]
            sample["candidates"] = r.get("pos_candidates", [])
            sample["oracle_action"] = self._extract_oracle_action(r["operation"])
            sample["metadata"] = {
                "website": r.get("website"),
                "domain": r.get("domain"),
            }
            results.append(sample)

        return results