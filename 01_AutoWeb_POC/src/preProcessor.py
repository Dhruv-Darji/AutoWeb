import json
import re
from io import BytesIO
from typing import List, Dict, Optional, Tuple

from PIL import Image
from bs4 import BeautifulSoup

class Preprocessor:
    """
    Minimal preprocessor: does NOT resize or OCR by default.
    It parses DOM and normalizes pos_candidates into structured entries.
    Returns a dict suitable for downstream steps.
    """
    def __init__(self, do_ocr: bool = False):
        self.do_ocr = do_ocr
    
    def _safe_load_candidate(self, c) -> dict:
        """
        Ensure candidate is a dict. If it's a JSON string, parse it.
        """
        if isinstance(c, dict):
            return c
        if isinstance(c, str):
            try:
                return json.loads(c)
            except Exception:
                # last resort: try eval-like fallback (not recommended)
                try:
                    return eval(c)
                except Exception:
                    return {"raw": c}
        return {"raw": c}

    def _parse_bounding_box_from_attrs(self, attrs_str: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Parse bounding_box_rect if present in attribute string.
        Returns (x1, y1, x2, y2) or None.
        """
        try:
            d = json.loads(attrs_str)
            rect = d.get("bounding_box_rect") or d.get("bounding_box")
            if rect:
                parts = [float(x) for x in str(rect).split(",")]
                if len(parts) == 4:
                    x, y, w, h = parts
                    return (x, y, x + w, y + h)

        except Exception:
            pass

        m = re.search(r'"bounding_box_rect"\s*:\s*"([^"]+)"', str(attrs_str))
        if not m:
            return None
        rect_str = m.group(1)
        try:
            x, y, w, h = [float(x.strip()) for x in rect_str.split(",")]
            return (x, y, x + w, y + h)
        except Exception:
            return None



    def parse_pos_candidates(self, pos_candidates: List, image_meta: Optional[dict] = None) -> List[Dict]:
        """
        Normalize 'pos_candidates' entries into a list of dicts:
        {
            'tag': ...,
            'bbox': (x1,y1,x2,y2) or None,
            'is_original_target': bool,
            'is_top_level_target': bool,
            'backend_node_id': ...,
            'raw': original_dict
        }
        Note: bbox coordinates are not scaled here (since we're not resizing).
        If `image_meta` contains scale_x/scale_y, they will be applied.
        """
        out = []
        for c in pos_candidates or []:
            cobj = self._safe_load_candidate(c)
            attrs = cobj.get("attributes", None)
            bbox = None

            if isinstance(attrs, str):
                bbox = self._parse_bounding_box_from_attrs(attrs)

            elif isinstance(attrs, dict):
                rect = attrs.get("bounding_box_rect") or attrs.get("bounding_box")
                if rect:
                    try:
                        parts = [float(x) for x in str(rect).split(",")]
                        x, y, w, h = parts
                        bbox = (x, y, x + w, y + h)
                    except Exception:
                        bbox = None
            
            # apply scaling if provided
            if bbox and image_meta:
                sx = image_meta.get("scale_x", 1.0)
                sy = image_meta.get("scale_y", 1.0)
                bbox = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)
            out.append({
                "tag": cobj.get("tag"),
                "bbox": bbox,
                "is_original_target": cobj.get("is_original_target", False),
                "is_top_level_target": cobj.get("is_top_level_target", False),
                "backend_node_id": cobj.get("backend_node_id"),
                "raw": cobj
            })
        return out

            


    def parse_dom_elements(self, html:Optional[str], max_elements: int = 500) -> List[Dict]:
        """
        Extract simple DOM element info from HTML using BeautifulSoup.
        Returns list of {tag, text, id, class, attrs, raw_html}.
        """
        out = []
        if not html:
            return out
        
        soup = BeautifulSoup(html, "html.parser")

        tags = ["a", "button", "input", "label", "select", "textarea", "div", "li", "span"]

        try:
            elems = soup.find_all(tags, limit=max_elements)
        except Exception:
            elems = soup.find_all(limit=max_elements)
        
        for t in elems:
            try:
                text = (t.get_text(separator=" ", strip=True) or "")[:200]
                eid = t.get("id")
                clazz = " ".join(t.get("class")) if t.get("class") else None
                attrs = dict(t.attrs)
                out.append({
                    "tag": t.name,
                    "text": text,
                    "id": eid,
                    "class": clazz,
                    "attrs": attrs,                    
                })
            except Exception:
                continue
        return out

    def run_ocr_stub(self, pil_image: Image.Image):
        return NotImplementedError("OCR not implemented in this stub.")

    def process(
            self,
            pil_image: Optional[Image.Image] = None,
            cleaned_html: Optional[str] = None,
            pos_candidates: Optional[List] = None,
    ) -> Dict:
        """
        Args:
            pil_image: Optional PIL.Image 
            cleaned_html: optional cleaned HTML
            pos_candidates: dataset pos_candidates column (list)
        Returns:
            {                   
                "dom_elements": [...],
                "candidates": [...],  # normalized candidates with bbox
                "ocr_tokens": [...],  # empty list if OCR disabled                
            }
        """

        # Parse DOM
        dom = None
        if cleaned_html:
            dom = self.parse_dom_elements(cleaned_html)
        else:
            dom = []

        # normalize candidates (no scaling since we are not resizing)
        candidates = self.parse_pos_candidates(pos_candidates or [], image_meta=None)

        ocr_tokens = []
        if self.do_ocr:
            if not pil_image:
                raise ValueError("OCR requested but no image provided.")
            ocr_tokens = self.run_ocr_stub(pil_image)
        
        return {
            "dom_elements": dom,
            "candidates": candidates,
            "ocr_tokens": ocr_tokens,
        }
