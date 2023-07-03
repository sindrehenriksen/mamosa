# MArkov MOdel Seismic Autotracking (mamosa)

## About
This code base was developed for my Master's thesis in Applied Mathematics. It comprises code to create synthetic seismic data volumes and three distinct seismic horizon autotrackers, two of which are based on a Hidden Markov Model.

The Master's thesis can be found in the project wiki section.

## Installation instructions
Dependencies for using the repo:
`pip install -r requirements.txt`

For developing:
`pip install -r dev-requirements.txt`

## Compiling new dependencies
Add depencies to `requirements.in` or `dev-requirements.in`, then run, in the following order,
```
pip install pip-tools  # if not installed
pip-compile requirements.in
pip-compile dev-requirements.in
```

## Pre-commit and -push hooks
Install git hooks that run linting and tests before commit/push:
```
pre-commit install -t pre-commit
pre-commit install -t pre-push
pip install gitlint
gitlint install-hook
```
