import pandas as pd
from pathlib import Path

class BrandFacts:
    def __init__(self, csv_path: str | Path):
        self.df = pd.read_csv(csv_path).fillna('')

    def get_brand_facts(self, brand_id: str, query: str = '', as_of: str = '2026-10-01') -> list[dict]:
        rows = self.df[self.df.brand_id.astype(str).str.lower() == brand_id.lower()].copy()
        if not rows.empty and 'valid_from' in rows:
            rows = rows[(rows.valid_from.astype(str) == '') | (rows.valid_from.astype(str) <= as_of)]
            rows = rows[(rows.valid_to.astype(str) == '') | (rows.valid_to.astype(str) >= as_of)]
        # Search is intentionally broad and deterministic; never hide facts when query matching is uncertain.
        if query.strip() and not rows.empty:
            terms = [t.lower() for t in query.split() if len(t) > 2]
            text = (rows['field'].astype(str) + ' ' + rows['value'].astype(str) + ' ' + rows['condition'].astype(str) + ' ' + rows['model_id'].astype(str)).str.lower()
            mask = text.apply(lambda s: any(t in s for t in terms))
            filtered = rows[mask]
            if len(filtered): rows = filtered
        return rows.to_dict(orient='records')
