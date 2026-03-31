"""Graph-based retrieval: given a query, find relevant articles via entity paths."""
from __future__ import annotations
import re
from .store import KGStore


class KGRetriever:
    """
    Retrieve source citations that are relevant to a query by:
    1. Detecting entity mentions in the query text.
    2. Walking the graph to collect related citations (1-2 hops).
    3. Ranking by hop distance and edge confidence.
    """

    def __init__(self, store: KGStore, max_hops: int = 2):
        self.store = store
        self.max_hops = max_hops

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Return a ranked list of {'citation': ..., 'score': ..., 'path': ...}
        for articles related to entity mentions found in the query.
        """
        mentions = self._detect_mentions(query)
        if not mentions:
            return []

        scores: dict[str, float] = {}
        paths: dict[str, str] = {}

        for mention in mentions:
            canonical = self.store.resolve(mention) or mention
            if not self.store.graph.has_node(canonical):
                continue
            self._bfs(canonical, scores, paths)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {'citation': cit, 'score': score, 'path': paths.get(cit, '')}
            for cit, score in ranked[:top_k]
        ]

    def _bfs(
        self,
        start: str,
        scores: dict[str, float],
        paths: dict[str, str],
        depth: int = 0,
    ) -> None:
        if depth > self.max_hops:
            return
        for _, neighbor, data in self.store.graph.out_edges(start, data=True):
            citation = data.get('source', '')
            if citation:
                decay = 1.0 / (2 ** depth)
                conf = data.get('confidence', 1.0)
                score = conf * decay
                if citation not in scores or scores[citation] < score:
                    scores[citation] = score
                    rel = data.get('relation', '')
                    paths[citation] = f'{start} --[{rel}]--> {neighbor}'
            if depth < self.max_hops:
                self._bfs(neighbor, scores, paths, depth + 1)

    def _detect_mentions(self, text: str) -> list[str]:
        """
        Find entity surface forms (canonical names or aliases) in the query.
        Uses a simple token-match against the store's entity and alias index.
        """
        found = []
        # Collect all known surface forms
        names = set(self.store.graph.nodes())
        aliases = {
            row[0]
            for row in self.store.conn.execute('SELECT alias FROM entity_aliases')
        }
        all_forms = names | aliases

        # Match longest forms first (greedy left-to-right)
        sorted_forms = sorted(all_forms, key=len, reverse=True)
        remaining = text
        for form in sorted_forms:
            if re.search(re.escape(form), remaining, re.IGNORECASE):
                found.append(form)
                remaining = re.sub(re.escape(form), '', remaining, flags=re.IGNORECASE)
        return found

    def entity_subgraph(self, name: str, hops: int = 1):
        """Return a NetworkX subgraph centred on the given entity."""
        import networkx as nx
        canonical = self.store.resolve(name) or name
        if not self.store.graph.has_node(canonical):
            return nx.MultiDiGraph()

        nodes = {canonical}
        frontier = {canonical}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.store.graph.successors(node))
                next_frontier.update(self.store.graph.predecessors(node))
            frontier = next_frontier - nodes
            nodes |= frontier

        return self.store.graph.subgraph(nodes).copy()
