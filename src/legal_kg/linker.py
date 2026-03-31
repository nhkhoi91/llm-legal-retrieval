"""Entity linker: normalise surface forms and detect abbreviation equivalences."""
from __future__ import annotations
import re
from .store import KGStore


# Known Swiss-law abbreviations that may appear bare in triples
_KNOWN_ABBREVS: dict[str, str] = {
    'BV': 'Bundesverfassung',
    'OR': 'Obligationenrecht',
    'ZGB': 'Zivilgesetzbuch',
    'StGB': 'Strafgesetzbuch',
    'DBG': 'Bundesgesetz über die direkte Bundessteuer',
    'MWSTG': 'Mehrwertsteuergesetz',
    'AHV': 'Alters- und Hinterlassenenversicherung',
    'AHVG': 'Bundesgesetz über die Alters- und Hinterlassenenversicherung',
    'IVG': 'Bundesgesetz über die Invalidenversicherung',
    'KVG': 'Bundesgesetz über die Krankenversicherung',
    'UVG': 'Bundesgesetz über die Unfallversicherung',
    'BVG': 'Bundesgesetz über die berufliche Alters-, Hinterlassenen- und Invalidenvorsorge',
    'FZG': 'Freizügigkeitsgesetz',
    'FINMA': 'Eidgenössische Finanzmarktaufsicht',
    'BAZL': 'Bundesamt für Zivilluftfahrt',
    'ASTRA': 'Bundesamt für Strassen',
    'BAFU': 'Bundesamt für Umwelt',
    'BFE': 'Bundesamt für Energie',
    'BAG': 'Bundesamt für Gesundheit',
    'SECO': 'Staatssekretariat für Wirtschaft',
    'EJPD': 'Eidgenössisches Justiz- und Polizeidepartement',
    'EFD': 'Eidgenössisches Finanzdepartement',
    'EDI': 'Eidgenössisches Departement des Innern',
    'UVEK': 'Eidgenössisches Departement für Umwelt, Verkehr, Energie und Kommunikation',
    'VBS': 'Eidgenössisches Departement für Verteidigung, Bevölkerungsschutz und Sport',
    'EDA': 'Eidgenössisches Departement für auswärtige Angelegenheiten',
    'WBF': 'Eidgenössisches Departement für Wirtschaft, Bildung und Forschung',
}

# Regex: "Vollname (ABK)" inline definition
_INLINE_ABBREV = re.compile(
    r'([A-ZÄÖÜ][^\(\)]{5,80}?)\s+\(([A-ZÄÖÜa-z][A-Za-z0-9\-]{1,15})\)'
)


class EntityLinker:
    """
    Normalises entity names in triples and registers abbreviation equivalences
    in the KGStore.

    Usage:
        linker = EntityLinker(store)
        linker.seed_known_abbrevs()          # load static dict
        linker.link_result(extraction_result)  # for each extraction
    """

    def __init__(self, store: KGStore):
        self.store = store

    def seed_known_abbrevs(self) -> None:
        """Register all statically known Swiss-law abbreviations."""
        for short, full in _KNOWN_ABBREVS.items():
            self.store._ensure_abbrev_link(short, full)

    def link_result(self, result) -> None:
        """
        Post-process one ExtractionResult:
        1. Register extracted abbreviations in the store.
        2. For each triple, resolve subject/object to canonical form.
        3. Patch triple in-place so add_result uses canonical names.
        """
        # Register abbreviations discovered in this article
        for short, full in result.abbreviations:
            self.store._ensure_abbrev_link(short, full)

        # Resolve triple endpoints to canonical names
        for triple in result.triples:
            canon_subj = self.store.resolve(triple.subject)
            canon_obj = self.store.resolve(triple.object)
            if canon_subj:
                triple.subject = canon_subj
            if canon_obj:
                triple.object = canon_obj

    def extract_inline_abbrevs(self, text: str) -> list[tuple[str, str]]:
        """
        Extract inline abbreviation definitions from raw text.
        Returns list of (short, full) pairs.
        Useful for pre-processing before extraction.
        """
        found: dict[str, tuple[str, str]] = {}
        for full, short in _INLINE_ABBREV.findall(text):
            full, short = full.strip(), short.strip()
            if short.isupper() or re.match(r'^[A-Z][a-z]?[A-Z]', short):
                found[short] = (short, full)
        return list(found.values())

    def merge_equivalent_entities(self) -> int:
        """Delegate to store's merge implementation."""
        return self.store.merge_equivalent_entities()
