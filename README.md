# Gymnasium Demo

## Setup using Pyenv and venv

```bash
pyenv local 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install 'gymnasium[all]' wasmtime
```

## Run

```bash
make run
```
