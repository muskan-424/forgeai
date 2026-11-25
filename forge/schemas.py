"""Typed contracts shared by every stage."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- documents ----------
class Chunk(BaseModel):
    id: str
    doc: str
    text: str
    heading: str = ""


# ---------- agent definition ----------
class RetrievalConfig(BaseModel):
    chunk_size: int = 800          # characters
    chunk_overlap: int = 100
    top_k: int = 4
    bm25_weight: float = 0.5       # 1.0 = BM25 only, 0.0 = embeddings only
    use_embeddings: bool = False
    min_score: float = 0.0         # drop chunks below this fused score (0..1)


class ToolRef(BaseModel):
    name: str
    justification: str = ""        # requirement that justifies attaching this tool
    config: dict = Field(default_factory=dict)  # e.g. {"csv_path": "docs/parts.csv"}


class OutputContract(BaseModel):
    """Every agent answers in this JSON shape; it is validated at runtime."""
    require_citations: bool = True
    max_answer_words: int = 150


class AgentDefinition(BaseModel):
    name: str
    description: str = ""
    model: str = ""                # empty => config.AGENT_MODEL
    temperature: float = 0.1
    system_prompt: str
    rules: list[str] = Field(default_factory=list)   # individually patchable prompt rules
    docs_dir: str = "docs"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: list[ToolRef] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)
    refusal_message: str = "I can't help with that; it is outside what my documents cover."


class Citation(BaseModel):
    doc: str = ""
    quote: str = ""


class AgentAnswer(BaseModel):
    """The output contract the agent must satisfy."""
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""


class AgentTrace(BaseModel):
    """Full record of one agent turn - used by judge and critic."""
    question: str
    retrieved: list[Chunk] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    raw_output: str = ""
    answer: AgentAnswer = Field(default_factory=AgentAnswer)
    contract_ok: bool = True
    citations_verified: list[bool] = Field(default_factory=list)
    latency_s: float = 0.0
    error: str = ""


# ---------- requirements (milestone 2+) ----------
class Requirements(BaseModel):
    brief: str
    scope: str = ""
    audience: str = ""
    tone: str = ""
    must_answer: list[str] = Field(default_factory=list)
    must_refuse: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarifications: list[dict] = Field(default_factory=list)  # {"q":..., "a":...}
    groundability: list[dict] = Field(default_factory=list)   # {"topic","status","evidence"}


# ---------- evals (milestone 2+) ----------
CaseType = Literal["answerable", "multi_doc", "out_of_scope", "ambiguous"]


class EvalCase(BaseModel):
    id: str
    type: CaseType
    question: str
    expected_facts: list[str] = Field(default_factory=list)
    source_docs: list[str] = Field(default_factory=list)
    source_quotes: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    split: Literal["dev", "holdout"] = "dev"
    notes: str = ""


class JudgeVerdict(BaseModel):
    case_id: str
    grounded: bool = True
    correct: bool = False
    refusal_correct: bool = True
    passed: bool = False
    fact_hits: list[bool] = Field(default_factory=list)
    fact_methods: list[str] = Field(default_factory=list)  # exact | fuzzy | llm | miss | uncertain
    uncertain: bool = False
    llm_used: bool = False
    reasons: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    case: EvalCase
    trace: AgentTrace
    verdict: JudgeVerdict
