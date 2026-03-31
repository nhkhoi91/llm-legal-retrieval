"""Build the legal knowledge graph from the Swiss-law corpus.

Flow per article:
  1. Extractor (Gemma) produces triples
  2. Judge reviews — if accept: write to store
                     if reject: Extractor retries with feedback
  3. Judge reviews retry — if accept: write to store
                           if reject: write to traces_review.jsonl for manual inspection

Usage:
    # Full run on Scaleway/Kaggle GPU:
    python scripts/build_kg.py --corpus data/laws_de.csv

    # Quick quality check (500 articles):
    python scripts/build_kg.py --corpus data/laws_de.csv --sample-seq 250 --sample-rand 250

    # Local trial (no GPU needed):
    python scripts/build_kg.py --corpus data/laws_de.csv --mock --sample-seq 20 --sample-rand 0
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from legal_kg.store import KGStore
from legal_kg.linker import EntityLinker
from legal_kg.models import ExtractionResult


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build legal knowledge graph')
    p.add_argument('--corpus', default='data/laws_de.csv')
    p.add_argument('--db-dir', default='data/kg')
    p.add_argument('--sample-seq', type=int, default=500,
                   help='First N rows (sequential by SR number)')
    p.add_argument('--sample-rand', type=int, default=500,
                   help='Random N rows from the remainder')
    p.add_argument('--full', action='store_true',
                   help='Process entire corpus (overrides --sample-seq/rand)')
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--no-4bit', action='store_true')
    p.add_argument('--mock', action='store_true',
                   help='Use stub extractor (no GPU needed, for local testing)')
    return p.parse_args()


# ── Mock extractor ────────────────────────────────────────────────────────────

def _mock_extract(citation: str, title: str, text: str,
                  feedback: str = '') -> tuple[ExtractionResult, str]:
    from legal_kg.models import Entity, Triple
    from legal_kg.extractor import _extract_abbrevs_regex
    words = [w for w in title.split() if len(w) > 3][:6]
    if len(words) < 2:
        words = ['Behörde', 'Verfahren', 'Antrag']
    entities = [Entity(name=w, type='concept') for w in words[:3]]
    triples = []
    relations = ['requires', 'regulates', 'defines', 'applies_to']
    for i in range(min(2, len(words) - 1)):
        triples.append(Triple(
            subject=words[i], relation=relations[i % len(relations)],
            relation_explain=f'Mock: text mentions {words[i]}',
            object=words[i + 1], source_citation=citation,
            confidence=round(random.uniform(0.75, 1.0), 2),
        ))
    abbrevs = _extract_abbrevs_regex(title + ' ' + text)
    raw = json.dumps({'mock': True, 'feedback_received': bool(feedback)})
    return ExtractionResult(citation=citation, entities=entities, triples=triples,
                            abbreviations=abbrevs, new_relations=[]), raw


def _mock_judge(citation: str, text: str,
                result: ExtractionResult) -> tuple[str, list, str, str]:
    # Accept everything in mock mode
    return 'accept', [], '', '{"verdict":"accept","issues":[],"feedback":""}'


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load corpus ───────────────────────────────────────────────────────────
    print(f'Loading corpus: {args.corpus}')
    df = pd.read_csv(args.corpus, low_memory=False)
    df[['citation', 'title', 'text']] = df[['citation', 'title', 'text']].fillna('')
    print(f'  Total rows: {len(df):,}')

    if args.full:
        df_sample = df
    else:
        first_n  = df.iloc[:args.sample_seq]
        rest     = df.iloc[args.sample_seq:]
        rand_n   = rest.sample(min(args.sample_rand, len(rest)), random_state=42)
        df_sample = pd.concat([first_n, rand_n]).reset_index(drop=True)
        print(f'  Sample: {len(first_n)} sequential + {len(rand_n)} random = {len(df_sample)}')

    rows = df_sample[['citation', 'title', 'text']].to_dict('records')

    # ── Init store + linker ───────────────────────────────────────────────────
    db_dir = Path(args.db_dir)
    store  = KGStore(db_dir)
    linker = EntityLinker(store)
    linker.seed_known_abbrevs()

    relations_path = db_dir / 'relations.json'
    traces_path    = db_dir / 'traces.jsonl'
    review_path    = db_dir / 'traces_review.jsonl'

    # ── Load persisted relations from previous run ────────────────────────────
    from legal_kg.extractor import RELATIONS
    if relations_path.exists():
        RELATIONS.update(json.loads(relations_path.read_text(encoding='utf-8')))
        print(f'  Loaded {len(RELATIONS)} relations from {relations_path}')
    else:
        print(f'  Starting with {len(RELATIONS)} seed relations')

    # ── Resume: skip already-processed citations ──────────────────────────────
    done: set[str] = set()
    if traces_path.exists():
        for line in traces_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                done.add(json.loads(line)['citation'])
    if review_path.exists():
        for line in review_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                done.add(json.loads(line)['citation'])
    rows_to_process = [r for r in rows if r.get('citation', '') not in done]
    print(f'  Already done: {len(done)}, remaining: {len(rows_to_process)}')

    if not rows_to_process:
        print('Nothing to do.')
        store.close()
        return

    # ── Init model ────────────────────────────────────────────────────────────
    if args.mock:
        print('Using mock extractor (no model loaded)')
        extract_fn = _mock_extract
        judge_fn   = _mock_judge
    else:
        from legal_kg.extractor import Extractor
        from legal_kg.judge import Judge
        extractor  = Extractor(load_in_4bit=not args.no_4bit)
        judge      = Judge(extractor)
        extract_fn = extractor.extract
        judge_fn   = judge.review

    # ── Extraction loop ───────────────────────────────────────────────────────
    traces_file = traces_path.open('a', encoding='utf-8')
    review_file = review_path.open('a', encoding='utf-8')

    stats = {'accept_r1': 0, 'accept_r2': 0, 'rejected': 0}

    for i, row in enumerate(rows_to_process):
        citation = row.get('citation', '')
        title    = str(row.get('title', '') or '')
        text     = str(row.get('text', '') or '')

        # ── Round 1: extract ──────────────────────────────────────────────────
        result1, raw1 = extract_fn(citation=citation, title=title, text=text)
        verdict1, issues1, feedback1, raw_judge1 = judge_fn(citation, text, result1)

        if verdict1 == 'accept':
            final_result = result1
            outcome = 'accept_r1'
            round2  = None
        else:
            # ── Round 2: retry with feedback ──────────────────────────────────
            result2, raw2 = extract_fn(citation=citation, title=title, text=text,
                                       feedback=feedback1)
            verdict2, issues2, feedback2, raw_judge2 = judge_fn(citation, text, result2)

            if verdict2 == 'accept':
                final_result = result2
                outcome = 'accept_r2'
            else:
                final_result = None
                outcome = 'rejected'

            round2 = {
                'raw_output': raw2,
                'entities':  [{'name': e.name, 'type': e.type} for e in result2.entities],
                'triples':   [{'subject': t.subject, 'relation': t.relation,
                               'relation_explain': t.relation_explain,
                               'object': t.object, 'confidence': t.confidence}
                              for t in result2.triples],
                'judge_verdict': verdict2,
                'judge_issues':  issues2,
                'judge_feedback': feedback2,
                'raw_judge':  raw_judge2,
            }

        # ── Register new relations discovered in accepted result ──────────────
        if final_result:
            for nr in final_result.new_relations:
                if nr['name'] not in RELATIONS:
                    RELATIONS[nr['name']] = nr['definition']
                    print(f'  [new relation] {nr["name"]}: {nr["definition"]}')
            linker.link_result(final_result)
            store.add_result(final_result)

        stats[outcome] += 1

        # ── Build trace ───────────────────────────────────────────────────────
        trace = {
            'citation':    citation,
            'title':       title[:200],
            'text_input':  text[:500],
            'outcome':     outcome,
            'round1': {
                'raw_output': raw1,
                'entities':  [{'name': e.name, 'type': e.type} for e in result1.entities],
                'triples':   [{'subject': t.subject, 'relation': t.relation,
                               'relation_explain': t.relation_explain,
                               'object': t.object, 'confidence': t.confidence}
                              for t in result1.triples],
                'new_relations': result1.new_relations,
                'judge_verdict': verdict1,
                'judge_issues':  issues1,
                'judge_feedback': feedback1,
                'raw_judge':     raw_judge1,
            },
        }
        if round2:
            trace['round2'] = round2

        # Write rejected to separate file for inspection; accepted to main traces
        target = review_file if outcome == 'rejected' else traces_file
        target.write(json.dumps(trace, ensure_ascii=False) + '\n')

        if (i + 1) % 10 == 0:
            traces_file.flush()
            review_file.flush()
            total_done = sum(stats.values())
            print(f'  {i+1}/{len(rows_to_process)}  '
                  f'entities={len(store._entities)}  '
                  f'edges={store.graph.number_of_edges()}  '
                  f'relations={len(RELATIONS)}  '
                  f'[r1={stats["accept_r1"]} r2={stats["accept_r2"]} rej={stats["rejected"]}]')

        if (i + 1) % args.save_every == 0:
            store.save()
            relations_path.write_text(
                json.dumps(RELATIONS, ensure_ascii=False, indent=2), encoding='utf-8')
            print('  [checkpoint saved]')

    traces_file.close()
    review_file.close()

    # ── Post-process ──────────────────────────────────────────────────────────
    n_merges = linker.merge_equivalent_entities()
    print(f'\nMerged {n_merges} duplicate entities')
    store.close()

    print(f'\nDone. outcomes: {stats}')
    kgs = store.stats()
    print('KG stats:')
    for k, v in kgs.items():
        print(f'  {k}: {v:,}')
    print(f'\nFinal relations ({len(RELATIONS)}):')
    for name, defn in RELATIONS.items():
        print(f'  {name}: {defn[:80]}')


if __name__ == '__main__':
    main()
