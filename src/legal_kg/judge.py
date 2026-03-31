"""LLM-as-judge: reviews extracted triples and returns accept/reject + feedback."""
from __future__ import annotations
import json
import re
import torch
from .models import ExtractionResult

JUDGE_PROMPT = """You are auditing legal knowledge graph triples extracted from a Swiss law article.

Citation: {citation}
Text: {text}

Extracted triples:
{triples_json}

For each triple check:
1. Subject and object are SPECIFIC legal terms actually present in the text
   (not generic words like "Recht", "Person", "Schaden", "Behörde").
2. Relation type is correct:
   - issued_by: ONLY if an AUTHORITY (Bundesrat, FINMA, Parlament) enacted the subject.
     WRONG: article -> parent law.
   - cites: ONLY if a DIFFERENT LAW is explicitly named by abbreviation (OR, ZGB, StGB...).
     WRONG: internal article refs, entity names, concepts.
   - regulates: subject governs/sets rules for object.
     WRONG: used as a catch-all fallback.
3. relation_explain quotes actual text from the article.

Return ONLY valid JSON, no markdown:
{{
  "verdict": "accept" or "reject",
  "issues": [
    {{"index": 0, "problem": "short description", "fix": "what to change"}}
  ],
  "feedback": "One short paragraph of specific actionable feedback referencing the actual text."
}}

Reject ONLY for serious errors (wrong relation type, hallucinated entities, internal refs mislabelled as cites).
Accept if triples are reasonable even if imperfect. Empty issues list and empty feedback on accept."""


class Judge:
    """Reuses the same loaded model as the Extractor to critique its output."""

    def __init__(self, extractor: object):
        # extractor must have .tokenizer and .model attributes
        self.tokenizer = extractor.tokenizer
        self.model = extractor.model

    def review(
        self,
        citation: str,
        text: str,
        result: ExtractionResult,
    ) -> tuple[str, list[dict], str, str]:
        """
        Returns (verdict, issues, feedback, raw_output).
          verdict  : 'accept' | 'reject'
          issues   : list of per-triple dicts {index, problem, fix}
          feedback : actionable text sent back to the extractor on retry
          raw_output: raw model response for logging
        """
        if not result.triples:
            return 'accept', [], '', ''

        triples_json = json.dumps(
            [
                {
                    'index': i,
                    'subject': t.subject,
                    'relation': t.relation,
                    'relation_explain': t.relation_explain,
                    'object': t.object,
                }
                for i, t in enumerate(result.triples)
            ],
            ensure_ascii=False,
            indent=2,
        )
        prompt_text = JUDGE_PROMPT.format(
            citation=citation,
            text=text[:1000],
            triples_json=triples_json,
        )
        messages = [{'role': 'user', 'content': prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(
            out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        verdict, issues, feedback = self._parse(raw)
        return verdict, issues, feedback, raw

    def _parse(self, raw: str) -> tuple[str, list[dict], str]:
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        raw = re.sub(r'\s*```$', '', raw.strip())
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return 'accept', [], ''
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return 'accept', [], ''
        verdict  = 'reject' if str(data.get('verdict', '')).lower() == 'reject' else 'accept'
        issues   = data.get('issues', []) or []
        feedback = (data.get('feedback') or '').strip()
        return verdict, issues, feedback
