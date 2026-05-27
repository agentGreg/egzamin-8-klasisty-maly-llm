import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOG = """
Iter 1: Val loss 1.327, Val took 2.5s
Iter 25: Train loss 0.6
Iter 25: Val loss 0.80, Val took 1.0s
Iter 50: Val loss 0.74, Val took 1.0s
Iter 75: Val loss 0.91, Val took 1.0s
"""


def _load():
    spec = importlib.util.spec_from_file_location("wc12", ROOT / "scripts" / "12_wybierz_checkpoint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_parse_val_losses():
    m = _load()
    assert m.parse_val_losses(LOG) == {1: 1.327, 25: 0.80, 50: 0.74, 75: 0.91}


def test_best_iter_ignores_iter1_warmup():
    m = _load()
    # iter 1 is the pre-training baseline; exclude it from selection
    assert m.best_iter({1: 1.327, 25: 0.80, 50: 0.74, 75: 0.91}) == 50
