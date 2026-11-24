#!/usr/bin/env python3
###############################################################################
# run.py
# Irrigation Technology Adoption Replicator-Dynamics Model
# Author: Yoshi Budiyanto, Arizona State University
# SPDX-FileCopyrightText: 2022-2026 Yoshi Budiyanto
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce Study 2 calculations and figures.')
    parser.add_argument('--numerical-only', action='store_true',
                        help='Generate calculations and 16 numerical figures; omit the two TeX figures.')
    parser.add_argument('--slice-python', type=Path, required=True,
                        help='Python interpreter with requirements-slices.txt installed.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    slice_python = args.slice_python.absolute()
    if not slice_python.is_file():
        parser.error('The slice Python interpreter does not exist.')
    if not args.numerical_only and shutil.which('pdflatex') is None:
        parser.error('pdflatex is required. Install TeX Live or use --numerical-only.')
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with tempfile.TemporaryDirectory(prefix='study2-') as temporary:
        env['MPLCONFIGDIR'] = str(Path(temporary) / 'matplotlib')
        for name in ('calculations.py', 'surface.py', 'surface_with_heatmap.py',
                     'surface_slices.py', 'replicator_phaseplane.py'):
            interpreter = str(slice_python) if name == 'surface_slices.py' else sys.executable
            subprocess.run([interpreter, str(root / 'code' / name)],
                           cwd=root, env=env, check=True)
        if not args.numerical_only:
            for stem in ('model_overview', 'water_use_comparison'):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                     '-output-directory', temporary, str(root / 'code' / (stem + '.tex'))],
                    cwd=root, env=env, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
                if result.returncode:
                    print('\n'.join(result.stdout.splitlines()[-35:]), file=sys.stderr)
                    raise subprocess.CalledProcessError(result.returncode, result.args)
                shutil.copyfile(Path(temporary) / (stem + '.pdf'), root / 'output' / (stem + '.pdf'))
                print('Generated', stem, flush=True)


if __name__ == '__main__':
    main()
