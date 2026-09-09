import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Generate a Markdown table from measured results')
    p.add_argument('run', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.run / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        p.error('Run is incomplete')
    rows = json.loads((a.run / 'summary.json').read_text())
    lines = ['# Measured transfer results', '',
             'Rates are percentages. Intervals are 95% Wilson intervals conditional on fixed checkpoints; they do not measure training-seed variability.', '',
             '| Source | Target | Attack | Clean target accuracy | Source ASR | Pair transfer (95% CI) | Eligible n | Conditional transfer |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    def rate(m):
        return 'undefined' if m['rate'] is None else f"{100*m['rate']:.2f}"
    for r in rows:
        pair = r['pair_transfer']
        interval = '' if pair['rate'] is None else f" ({100*pair['ci95'][0]:.2f}, {100*pair['ci95'][1]:.2f})"
        lines.append(f"| {r['source']} | {r['target']} | {r['attack']} | {rate(r['target_clean_accuracy'])} | {rate(r['source_asr'])} | {rate(pair)}{interval} | {pair['total']} | {rate(r['conditional_transfer'])} |")
    (a.run / 'table.md').write_text('\n'.join(lines) + '\n')
    print(a.run / 'table.md')


if __name__ == '__main__':
    main()
