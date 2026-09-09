"""Small real-data execution check; never a final empirical benchmark."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from .common import environment, write_json
from .models import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', default='data')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', pilot=True, environment=environment(), stages=[])
    started = time.perf_counter()

    def run(name, module, *options):
        command = [sys.executable, '-m', 'transferlab.' + module, *map(str, options)]
        print(f'Pilot stage: {name}', flush=True)
        stage = dict(name=name, command=command)
        record['stages'].append(stage)
        write_json(args.output / 'pilot.json', record)
        stage_start = time.perf_counter()
        with (args.output / f'{name}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        stage.update(elapsed_seconds=time.perf_counter()-stage_start, returncode=result.returncode)
        write_json(args.output / 'pilot.json', record)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {args.output / (name + ".log")}')
        print(f'Completed {name} in {stage["elapsed_seconds"]:.1f}s', flush=True)

    try:
        checkpoints = []
        for architecture in NAMES:
            output = args.output / architecture
            run('train-' + architecture, 'train', '--architecture', architecture, '--output', output,
                '--data', args.data, '--device', args.device, '--epochs', 2,
                '--train-samples', 2048, '--validation-samples', 512, '--batch-size', 64)
            checkpoints.append(output / 'best.pt')
        common = ['--checkpoints', *checkpoints, '--data', args.data, '--device', args.device, '--batch-size', 32]
        for name, options in [
            ('linf', ['--samples', 256, '--attacks', 'clean', 'noise', 'fgsm', 'pgd', '--steps', 5, '--restarts', 1]),
            ('zero', ['--samples', 32, '--attacks', 'clean', 'noise', 'fgsm', 'pgd', '--epsilon', 0, '--steps', 2, '--restarts', 1]),
            ('cw', ['--samples', 8, '--attacks', 'cw', '--cw-steps', 5, '--cw-search', 1]),
        ]:
            output = args.output / name
            run('evaluate-' + name, 'evaluate', *common, '--output', output, *options)
            run('report-' + name, 'report', output)
        for row in json.loads((args.output / 'zero' / 'summary.json').read_text()):
            if row['max_linf'] != 0 or row['source_asr']['successes'] != 0 or row['pair_transfer']['successes'] != 0:
                raise RuntimeError('Zero-budget control failed')
        run('figure', 'figure', args.output / 'linf' / 'example.pt', '--output', args.output / 'linf' / 'example.png')
        record['status'] = 'complete'
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        write_json(args.output / 'pilot.json', record)
    print(f'Pilot complete in {record["elapsed_seconds"]:.1f}s. Results are execution checks only.', flush=True)


if __name__ == '__main__':
    main()
