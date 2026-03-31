"""Data models for the legal knowledge graph."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    type: str                          # law | concept | authority | person | procedure
    abbreviations: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


@dataclass
class Triple:
    subject: str
    relation: str
    relation_explain: str              # why this relation was chosen (quotes source text)
    object: str
    source_citation: str
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    citation: str
    entities: list[Entity]
    triples: list[Triple]
    abbreviations: list[tuple[str, str]]   # (short, full)
    new_relations: list[dict] = field(default_factory=list)  # [{"name": ..., "definition": ...}]
