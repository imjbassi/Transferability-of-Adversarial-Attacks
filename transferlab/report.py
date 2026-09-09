import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from .common import digest
from .metrics import summarize


def verified_results(run, manifest):
    predictions = run / 'predictions.csv'
    if digest(predictions) != manifest['predictions_sha256']:
        raise ValueError('Prediction checksum mismatch')
    groups = defaultdict(list)
    with predictions.open(newline='') as f:
        for row in csv.DictReader(f):
            for key in ['index', 'label', 'source_clean', 'target_clean', 'source_adv', 'target_adv']:
                row[key] = int(row[key])
            for key in ['linf', 'l2']:
                row[key] = float(row[key])
            groups[row['source'], row['target'], row['attack']].append(row)
    results = []
    expected_groups = {(s, t, attack) for s in manifest['checkpoints'] for t in manifest['checkpoints']
                       for attack in manifest['config']['attacks']}
    if set(groups) != expected_groups:
        raise ValueError('Prediction groups do not match the manifest')
    for (source, target, attack), rows in groups.items():
        if [r['index'] for r in rows] != manifest['test_indices']:
            raise ValueError('Prediction indices do not match the manifest')
        results.append(dict(source=source, target=target, attack=attack, **summarize(rows)))
    saved = json.loads((run / 'summary.json').read_text())
    # Older summaries lack max_l2; all other fields are required.
    expected = [{k: v for k, v in r.items() if k != 'max_l2' or 'max_l2' in old}
                for r, old in zip(results, saved)]
    if len(results) != len(saved) or expected != saved:
        raise ValueError('Summary does not match recomputed prediction counts')
    return results


def main():
    p = argparse.ArgumentParser(description='Generate a Markdown table from measured results')
    p.add_argument('run', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.run / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        p.error('Run is incomplete')
    rows = verified_results(a.run, manifest)
    lines = ['# Pilot execution results (not final findings)' if manifest.get('pilot') else '# Measured transfer results', '',
             'Rates are percentages. Intervals are 95% Wilson intervals conditional on fixed checkpoints; they do not measure training-seed variability.', '',
             'Every rate includes numerator/denominator. Undefined means no eligible examples. Norms are measured in raw pixels.', '',
             '| Source | Target | Attack | Clean source | Clean target | Attacked accuracy | Source ASR | Pair transfer (95% CI) | Conditional transfer | Mean L2 | Max L2 | Max Linf |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def rate(m):
        value = 'undefined' if m['rate'] is None else f"{100*m['rate']:.2f}"
        return f"{value} [{m['successes']}/{m['total']}]"
    for r in rows:
        pair = r['pair_transfer']
        interval = '' if pair['rate'] is None else f" ({100*pair['ci95'][0]:.2f}, {100*pair['ci95'][1]:.2f})"
        lines.append(f"| {r['source']} | {r['target']} | {r['attack']} | {rate(r['source_clean_accuracy'])} | {rate(r['target_clean_accuracy'])} | {rate(r['adversarial_accuracy'])} | {rate(r['source_asr'])} | {rate(pair)}{interval} | {rate(r['conditional_transfer'])} | {r['mean_l2']:.6f} | {r['max_l2']:.6f} | {r['max_linf']:.6f} |")
    (a.run / 'table.md').write_text('\n'.join(lines) + '\n')
    print(a.run / 'table.md')


if __name__ == '__main__':
    main()
