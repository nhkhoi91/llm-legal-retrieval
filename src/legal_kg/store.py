"""JSON-backed store + NetworkX graph for the legal knowledge graph.

Storage layout:
  <db_dir>/entities.json   -- {canonical_name: {"type": ..., "abbreviations": [...], "aliases": [...]}}
  <db_dir>/triples.jsonl   -- one JSON object per line
"""
from __future__ import annotations
import json
from pathlib import Path
import networkx as nx
from .models import Entity, Triple, ExtractionResult


class KGStore:
    """Persistent store for entities and triples with in-memory NetworkX graph."""

    def __init__(self, db_dir: str | Path = 'data/kg'):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._entities_path = self.db_dir / 'entities.json'
        self._triples_path = self.db_dir / 'triples.jsonl'

        # {canonical_name: {"type": str, "abbreviations": [str], "aliases": [str]}}
        self._entities: dict[str, dict] = {}
        # alias/abbrev → canonical name
        self._alias_index: dict[str, str] = {}

        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._entities_path.exists():
            raw = self._entities_path.read_text(encoding='utf-8')
            self._entities = json.loads(raw)
            for name, info in self._entities.items():
                self.graph.add_node(name, type=info['type'])
                for alias in info.get('aliases', []):
                    self._alias_index[alias] = name
                for abbrev in info.get('abbreviations', []):
                    self._alias_index[abbrev] = name

        if self._triples_path.exists():
            raw = self._triples_path.read_text(encoding='utf-8')
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                self.graph.add_edge(
                    t['subject'], t['object'],
                    relation=t['relation'],
                    relation_explain=t.get('relation_explain', ''),
                    source=t.get('source_citation', ''),
                    confidence=t.get('confidence', 1.0),
                )

    def save(self) -> None:
        """Flush in-memory state to disk."""
        self._entities_path.write_text(
            json.dumps(self._entities, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        lines = []
        for subj, obj, data in self.graph.edges(data=True):
            lines.append(json.dumps({
                'subject': subj,
                'relation': data.get('relation', ''),
                'relation_explain': data.get('relation_explain', ''),
                'object': obj,
                'source_citation': data.get('source', ''),
                'confidence': data.get('confidence', 1.0),
            }, ensure_ascii=False))
        self._triples_path.write_text(
            '\n'.join(lines), encoding='utf-8'
        )

    # ── Upsert helpers ────────────────────────────────────────────────────────

    def upsert_entity(self, entity: Entity) -> None:
        if entity.name not in self._entities:
            self._entities[entity.name] = {
                'type': entity.type,
                'abbreviations': [],
                'aliases': [],
            }
            self.graph.add_node(entity.name, type=entity.type)

        info = self._entities[entity.name]
        for alias in entity.aliases:
            if alias not in info['aliases']:
                info['aliases'].append(alias)
            self._alias_index[alias] = entity.name

        for abbrev in entity.abbreviations:
            if abbrev not in info['abbreviations']:
                info['abbreviations'].append(abbrev)
            self._alias_index[abbrev] = entity.name

    def add_triple(self, triple: Triple) -> None:
        # Ensure nodes exist (may be bare names not yet in entities dict)
        for name in (triple.subject, triple.object):
            if name not in self._entities:
                self._entities[name] = {
                    'type': 'concept', 'abbreviations': [], 'aliases': [],
                }
                self.graph.add_node(name, type='concept')

        self.graph.add_edge(
            triple.subject, triple.object,
            relation=triple.relation,
            relation_explain=triple.relation_explain,
            source=triple.source_citation,
            confidence=triple.confidence,
        )

    def add_result(self, result: ExtractionResult) -> None:
        for entity in result.entities:
            self.upsert_entity(entity)
        for short, full in result.abbreviations:
            self._ensure_abbrev_link(short, full)
        for triple in result.triples:
            self.add_triple(triple)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def resolve(self, name: str) -> str | None:
        """Return canonical name for any surface form."""
        if name in self._entities:
            return name
        return self._alias_index.get(name)

    def neighbors(self, name: str, relation: str | None = None) -> list[dict]:
        canonical = self.resolve(name) or name
        if not self.graph.has_node(canonical):
            return []
        results = []
        for _, tgt, data in self.graph.out_edges(canonical, data=True):  # noqa: E501
            if relation and data.get('relation') != relation:
                continue
            results.append({'object': tgt, **data})
        return results

    def entity_info(self, name: str) -> dict | None:
        canonical = self.resolve(name)
        if canonical is None:
            return None
        return {'name': canonical, **self._entities[canonical]}

    # ── Abbreviation linking ──────────────────────────────────────────────────

    def _ensure_abbrev_link(self, short: str, full: str) -> None:
        """Make short and full point to the same canonical entity."""
        full_canon = self.resolve(full)
        short_canon = self.resolve(short)

        if full_canon is not None:
            # full already exists → register short as its abbreviation
            info = self._entities[full_canon]
            if short not in info['abbreviations']:
                info['abbreviations'].append(short)
            self._alias_index[short] = full_canon
        elif short_canon is not None:
            # short already exists → register full as its alias
            info = self._entities[short_canon]
            if full not in info['aliases']:
                info['aliases'].append(full)
            self._alias_index[full] = short_canon
        else:
            # Neither exists → create with full as canonical
            self._entities[full] = {
                'type': 'law',
                'abbreviations': [short],
                'aliases': [],
            }
            self.graph.add_node(full, type='law')
            self._alias_index[short] = full

    # ── Merge duplicates ──────────────────────────────────────────────────────

    def merge_equivalent_entities(self) -> int:
        """
        If an alias is itself a canonical entity name, merge the duplicate
        into the primary (the one that owns the alias).
        Returns number of merges performed.
        """
        merges = 0
        for alias, primary in list(self._alias_index.items()):
            if alias == primary:
                continue
            if alias not in self._entities:
                continue
            # alias is also a canonical name → merge alias→primary
            duplicate = alias
            dup_info = self._entities.pop(duplicate)

            # Migrate aliases of the duplicate to primary
            primary_info = self._entities[primary]
            for a in dup_info.get('aliases', []):
                if a not in primary_info['aliases'] and a != primary:
                    primary_info['aliases'].append(a)
                self._alias_index[a] = primary
            for ab in dup_info.get('abbreviations', []):
                if ab not in primary_info['abbreviations'] and ab != primary:
                    primary_info['abbreviations'].append(ab)
                self._alias_index[ab] = primary

            # Rewire graph edges
            if self.graph.has_node(duplicate):
                for _, tgt, data in list(self.graph.out_edges(duplicate, data=True)):
                    self.graph.add_edge(primary, tgt, **data)
                for src, _, data in list(self.graph.in_edges(duplicate, data=True)):
                    self.graph.add_edge(src, primary, **data)
                self.graph.remove_node(duplicate)

            merges += 1
        return merges

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'entities': len(self._entities),
            'aliases': len(self._alias_index),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
        }

    def close(self) -> None:
        self.save()
