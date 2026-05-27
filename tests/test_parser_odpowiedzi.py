import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_05():
    spec = importlib.util.spec_from_file_location("ocen05", ROOT / "scripts" / "05_ocen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_strict_tag_letter():
    m = _load_05()
    assert m.wylusk_zamkniete("blah <odpowiedz>C</odpowiedz>", "C") == "C"


def test_clean_field_short_circuits():
    m = _load_05()
    assert m.wylusk_zamkniete("irrelevant text", "BD") == "BD"


def test_boxed_fallback():
    m = _load_05()
    assert m.wylusk_zamkniete(r"final \boxed{A}", "") == "A"


def test_gsm8k_hash_fallback():
    m = _load_05()
    assert m.wylusk_zamkniete("rozumowanie...\n#### D", "") == "D"


def test_enumerated_last_line():
    m = _load_05()
    assert m.wylusk_zamkniete("A. zle\nB. tez zle\nC. dobrze", "") == "C"


def test_normalize_strips_ws():
    m = _load_05()
    assert m.normalize(" a c ") == "AC"


def test_no_match_returns_field_upper():
    m = _load_05()
    assert m.wylusk_zamkniete("kompletnie bez litery", "") == ""
