import csv
from pathlib import Path

from .base import Manifest, Tool


class CsvLookup(Tool):
    manifest = Manifest(
        name="csv_lookup",
        description="Look up rows in the configured CSV table where a column equals or contains a value.",
        args={"column": "column name (omit to search all columns)", "value": "text to match (case-insensitive)"},
        when_needed="Brief involves structured tabular data (catalogue, price list, inventory) provided as a CSV document.",
        keywords=["catalogue", "catalog", "price", "part number", "stock", "inventory", "sku", "spare", "lookup", "table"],
        strong_keywords=["catalogue", "catalog", "price list", "look up", "lookup"],
        requires_file_type="csv",
    )

    def run(self, args: dict) -> str:
        rel = self.config.get("csv_path")
        if not rel:
            return "error: csv_path not configured"
        path = Path(self.docs_dir) / rel
        if not path.exists():
            return f"error: file {rel} not found"
        value = str(args.get("value", "")).strip().lower()
        col = str(args.get("column", "")).strip()
        if not value:
            return "error: value required"
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if col and rows and col not in rows[0]:
            return f"error: unknown column '{col}'. Columns: {', '.join(rows[0].keys())}"
        hits = [r for r in rows if (value in str(r.get(col, "")).lower() if col
                                    else any(value in str(v).lower() for v in r.values()))]
        if not hits:
            return "no matching rows"
        return "\n".join("; ".join(f"{k}: {v}" for k, v in r.items()) for r in hits[:5]) + \
               (f"\n(+{len(hits) - 5} more)" if len(hits) > 5 else "")
