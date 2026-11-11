from pathlib import Path
import argparse
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')

parser = argparse.ArgumentParser(description='Run or check the assignment notebooks in order.')
parser.add_argument('--check', action='store_true', help='Compile code cells without executing them.')
parser.add_argument('--question', choices=['1','2','3','4'], help='Run one notebook instead of all four.')
args = parser.parse_args()
questions = [int(args.question)] if args.question else [1, 2, 3, 4]

for number in questions:
    path = ROOT / f'Q{number}.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code_cells = [(i, ''.join(c['source'])) for i,c in enumerate(notebook['cells']) if c['cell_type'] == 'code']
    for i, source in code_cells:
        compile(source, f'{path.name}:cell-{i}', 'exec')
    print(f'{path.name}: {len(code_cells)} code cells compile', flush=True)
    if args.check:
        continue
    from IPython.display import display
    namespace = {'__name__': '__main__', 'display': display}
    print(f'Running {path.name}; existing results can be overwritten.', flush=True)
    for i, source in code_cells:
        print(f'\n--- {path.name}, cell {i} ---', flush=True)
        try:
            exec(compile(source, f'{path.name}:cell-{i}', 'exec'), namespace)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    print(f'Finished {path.name}', flush=True)
