"""Tool registry. The generator may only attach tools listed here."""
from .base import Manifest, Tool
from .calculator import Calculator
from .csv_lookup import CsvLookup
from .current_date import CurrentDate
from .open_meteo import OpenMeteo

REGISTRY: dict[str, type[Tool]] = {
    t.manifest.name: t for t in (CsvLookup, Calculator, OpenMeteo, CurrentDate)
}


def manifests() -> list[Manifest]:
    return [t.manifest for t in REGISTRY.values()]


def build_tool(name: str, config: dict | None = None, docs_dir: str = ".") -> Tool:
    if name not in REGISTRY:
        raise KeyError(f"unknown tool '{name}'. Registry: {sorted(REGISTRY)}")
    return REGISTRY[name](config, docs_dir)
