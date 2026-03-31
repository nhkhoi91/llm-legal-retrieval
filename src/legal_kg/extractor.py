"""Triple extractor using google/gemma-2-9b-it via transformers."""
from __future__ import annotations
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .models import Entity, Triple, ExtractionResult

MODEL_NAME = 'google/gemma-2-9b-it'

# ── Known relations — grows as Gemma discovers new ones ──────────────────────
# Injected into every prompt so the model knows what's available.
RELATIONS: dict[str, str] = {
    "requires":   "subject IMPOSES an obligation ON object. e.g. 'Arbeitgeber requires Beitragspflicht'",
    "regulates":  "subject GOVERNS or SETS RULES FOR object. e.g. 'KVG regulates Krankenversicherung'",
    "issued_by":  "subject (a rule/ordinance) was ENACTED BY object, which must be an AUTHORITY "
                  "(Bundesrat, Parlament, FINMA). "
                  "WRONG: article->law, law->canton. RIGHT: Ausfuehrungsbestimmungen->Bundesrat",
    "cites":      "subject EXPLICITLY REFERENCES a DIFFERENT LAW by name or abbreviation "
                  "(OR, ZGB, StGB, KVG...). "
                  "WRONG: pointing at an entity, person, or same-law article. "
                  "RIGHT: 'Schadensersatzanspruch cites Obligationenrecht' because text says 'gemaess OR'",
    "exempts":    "subject grants an exception or exclusion from object",
    "modifies":   "subject amends, replaces, or changes object",
    "defines":    "subject gives the legal definition of object. e.g. 'gilt als', 'im Sinne dieses Gesetzes'",
    "applies_to": "subject's scope, rule, or obligation covers object",
}


def _relations_block() -> str:
    return '\n'.join(f'- {name}: {defn}' for name, defn in RELATIONS.items())


PROMPT_TEMPLATE = """You are a legal knowledge graph builder for Swiss federal law (German).
Extract named entities and relationships from the article below.

Citation: {citation}
Title: {title}
Text: {text}

Entity types:
- law: a statute, ordinance, or treaty (e.g. "Obligationenrecht", "OR", "ZGB", "KVG")
- authority: an official body or government office (e.g. "Bundesrat", "FINMA", "Bundesgericht")
- concept: a specific legal term (e.g. "Versicherungspflicht", "Schadensersatzanspruch", "Verjährungsfrist")
- person: a legal role (e.g. "Arbeitnehmer", "Antragsteller", "Versicherter")
- procedure: a formal legal process (e.g. "Einsprache", "Bewilligung", "Volksinitiative")

Known relations:
{relations_block}

If NONE of the above relations fit, you may invent a new one. Add it to "new_relations" with its definition.

Each triple MUST include a "relation_explain" field: one sentence explaining WHY you chose this relation,
quoting the specific text that justifies it.

---
EXAMPLE 1 — issued_by (authority enacted the rule) + cites (cross-law reference)
Input: "Der Bundesrat erlässt die Ausführungsbestimmungen. Er kann dabei auf das Obligationenrecht (OR) verweisen."
Output:
{{
  "entities": [
    {{"name": "Bundesrat", "type": "authority", "abbreviations": [], "aliases": []}},
    {{"name": "Ausführungsbestimmungen", "type": "concept", "abbreviations": [], "aliases": []}},
    {{"name": "Obligationenrecht", "type": "law", "abbreviations": ["OR"], "aliases": []}}
  ],
  "triples": [
    {{
      "subject": "Ausführungsbestimmungen", "relation": "issued_by",
      "relation_explain": "The text says 'Der Bundesrat erlässt' — Bundesrat is the enacting authority.",
      "object": "Bundesrat", "confidence": 0.95
    }},
    {{
      "subject": "Ausführungsbestimmungen", "relation": "cites",
      "relation_explain": "The text explicitly references 'das Obligationenrecht (OR)', a different law.",
      "object": "Obligationenrecht", "confidence": 0.90
    }}
  ],
  "new_relations": [],
  "abbreviations": [{{"short": "OR", "full": "Obligationenrecht"}}]
}}

---
EXAMPLE 2 — cites bridges two laws
Input (Art. 955 ZGB): "Der Kanton haftet für den Schaden aus Fehlern des Grundbuchamtes. Der Schadensersatzanspruch verjährt gemäss den Vorschriften des Obligationenrechts (OR)."
Output:
{{
  "entities": [
    {{"name": "Kanton", "type": "authority", "abbreviations": [], "aliases": []}},
    {{"name": "Grundbuchamt", "type": "authority", "abbreviations": [], "aliases": []}},
    {{"name": "Haftung des Kantons", "type": "concept", "abbreviations": [], "aliases": []}},
    {{"name": "Schadensersatzanspruch", "type": "concept", "abbreviations": [], "aliases": []}},
    {{"name": "Obligationenrecht", "type": "law", "abbreviations": ["OR"], "aliases": []}}
  ],
  "triples": [
    {{
      "subject": "Kanton", "relation": "requires",
      "relation_explain": "The text says 'haftet' — the canton is obligated to bear liability.",
      "object": "Haftung des Kantons", "confidence": 0.95
    }},
    {{
      "subject": "Schadensersatzanspruch", "relation": "cites",
      "relation_explain": "'verjährt gemäss den Vorschriften des Obligationenrechts (OR)' explicitly names OR.",
      "object": "Obligationenrecht", "confidence": 0.95
    }}
  ],
  "new_relations": [],
  "abbreviations": [{{"short": "OR", "full": "Obligationenrecht"}}]
}}

---
EXAMPLE 3 — new relation invented
Input (Art. 60 OR): "Der Anspruch auf Schadensersatz verjährt mit Ablauf von einem Jahr. Bei unerlaubten Handlungen beträgt die absolute Verjährungsfrist zehn Jahre."
Output:
{{
  "entities": [
    {{"name": "Schadensersatzanspruch", "type": "concept", "abbreviations": [], "aliases": ["Anspruch auf Schadensersatz"]}},
    {{"name": "Verjährungsfrist", "type": "concept", "abbreviations": [], "aliases": []}},
    {{"name": "unerlaubte Handlung", "type": "concept", "abbreviations": [], "aliases": []}}
  ],
  "triples": [
    {{
      "subject": "Verjährungsfrist", "relation": "applies_to",
      "relation_explain": "The text sets a 1-year limitation period specifically for Schadensersatzanspruch.",
      "object": "Schadensersatzanspruch", "confidence": 0.95
    }},
    {{
      "subject": "unerlaubte Handlung", "relation": "triggers",
      "relation_explain": "'bei unerlaubten Handlungen beträgt die absolute Verjährungsfrist zehn Jahre' — the act triggers the extended period.",
      "object": "Verjährungsfrist", "confidence": 0.85
    }}
  ],
  "new_relations": [
    {{"name": "triggers", "definition": "subject (an event or act) causes or activates object (a legal consequence or period)"}}
  ],
  "abbreviations": []
}}

---
Now extract from the article above. Return ONLY valid JSON — no explanation, no markdown.
Rules:
- Only extract SPECIFIC legal terms — not generic words like "Schaden", "Person", "Recht"
- Use compound terms: "Schadensersatzanspruch" not "Schaden", "Verjährungsfrist" not "Frist"
- Canonical German nominative singular form
- issued_by: ONLY when an authority enacted something — never article->law
- cites: ONLY when text explicitly names a DIFFERENT law by abbreviation or full name
- relation_explain: required for every triple — quote the specific text that justifies it
- aliases: fill when the same concept appears under two phrasings in this text
- new_relations: only if no known relation fits; include name + definition
- Only include triples with confidence >= 0.7
- Empty lists if nothing to extract{feedback_block}"""


def _build_prompt(citation: str, title: str, text: str, feedback: str = '') -> str:
    feedback_block = ''
    if feedback:
        feedback_block = (
            '\n\nPREVIOUS ATTEMPT WAS REJECTED. Reviewer feedback:\n'
            + feedback
            + '\nPlease fix the issues above.'
        )
    return PROMPT_TEMPLATE.format(
        citation=citation,
        title=title[:300],
        text=text[:1500],
        relations_block=_relations_block(),
        feedback_block=feedback_block,
    )


def _parse(raw: str, citation: str, regex_abbrevs: list) -> ExtractionResult:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return ExtractionResult(citation=citation, entities=[], triples=[],
                                abbreviations=regex_abbrevs, new_relations=[])
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return ExtractionResult(citation=citation, entities=[], triples=[],
                                abbreviations=regex_abbrevs, new_relations=[])

    entities = [
        Entity(
            name=(e.get('name') or '').strip(),
            type=e.get('type') or 'concept',
            abbreviations=e.get('abbreviations') or [],
            aliases=e.get('aliases') or [],
        )
        for e in data.get('entities', [])
        if (e.get('name') or '').strip()
    ]
    triples = [
        Triple(
            subject=(t.get('subject') or '').strip(),
            relation=t.get('relation') or 'relates_to',
            relation_explain=(t.get('relation_explain') or '').strip(),
            object=(t.get('object') or '').strip(),
            source_citation=citation,
            confidence=float(t.get('confidence') or 1.0),
        )
        for t in data.get('triples', [])
        if (t.get('subject') or '').strip() and (t.get('object') or '').strip()
    ]
    llm_abbrevs = [
        (a['short'].strip(), a['full'].strip())
        for a in data.get('abbreviations', [])
        if (a.get('short') or '').strip() and (a.get('full') or '').strip()
    ]
    merged = {a[0]: a for a in regex_abbrevs + llm_abbrevs}
    new_relations = [
        {'name': r['name'].strip(), 'definition': r['definition'].strip()}
        for r in data.get('new_relations', [])
        if (r.get('name') or '').strip() and (r.get('definition') or '').strip()
    ]
    return ExtractionResult(
        citation=citation, entities=entities, triples=triples,
        abbreviations=list(merged.values()), new_relations=new_relations,
    )


def _extract_abbrevs_regex(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r'([A-ZÄÖÜ][^\(\)]{5,80}?)\s+\(([A-ZÄÖÜa-z][A-Za-z0-9\-]{1,15})\)')
    found = {}
    for full, short in pattern.findall(text):
        full, short = full.strip(), short.strip()
        if short.isupper() or re.match(r'^[A-Z][a-z]?[A-Z]', short):
            found[short] = (short, full)
    return list(found.values())


class Extractor:
    def __init__(self, load_in_4bit: bool = True):
        print(f'Loading {MODEL_NAME}...')
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        ) if load_in_4bit else None
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_cfg,
            torch_dtype=None if load_in_4bit else torch.bfloat16,
            device_map='auto',
        )
        self.model.eval()
        print('Model ready')

    def _run(self, prompt_text: str, max_new_tokens: int = 600) -> str:
        messages = [{'role': 'user', 'content': prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=self.tokenizer.eos_token_id)
        new_ids = out[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def extract(self, citation: str, title: str, text: str,
                feedback: str = '') -> tuple[ExtractionResult, str]:
        """Returns (result, raw_output)."""
        regex_abbrevs = _extract_abbrevs_regex(title + ' ' + text)
        raw = self._run(_build_prompt(citation, title, text, feedback))
        return _parse(raw, citation, regex_abbrevs), raw
