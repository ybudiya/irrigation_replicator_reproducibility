# Irrigation Replicator Reproducibility

Code, scenario parameters, and instructions to reproduce the calculations and figures for a three-strategy replicator-dynamics model of irrigation technology adoption.

The model compares gravity, pressurized, and computerized irrigation strategies through relative crop yields, production costs, and market conditions.

## Installation and execution

See [_administrative/README.md](_administrative/README.md) for installation, dependencies, execution instructions, and the figure and table mapping.

After installing the dependencies as described there, run from the repository root:

```sh
.venv/bin/python run.py --slice-python .venv-slices/bin/python
```

To run the numerical calculations and figures without the two LaTeX diagrams:

```sh
.venv/bin/python run.py --slice-python .venv-slices/bin/python --numerical-only
```

On Windows, replace `bin/python` with `Scripts/python.exe`.

## Contents

| Path | Purpose |
|---|---|
| `run.py` | Entry point for calculations and figure generation |
| `code/` | Analytical calculations and figure-generation code |
| `data/scenarios.csv` | Scenario parameters |
| `requirements.txt` | Dependencies for calculations, surfaces, heatmaps, and phase planes |
| `requirements-slices.txt` | Dependencies for the slice figures |
| `_administrative/` | Detailed instructions, source records, reference results, and verification |

Results are generated in `output/` when the code runs. The study-area map is outside the executable workflow; its source limitation is documented in `_administrative/README.md`.

## License

Licensed under the [MIT License](LICENSE).

Copyright (c) 2022-2026 Yoshi Budiyanto.
