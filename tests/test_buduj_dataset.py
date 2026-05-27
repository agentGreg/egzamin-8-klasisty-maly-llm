import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("bd08", ROOT / "scripts" / "08_buduj_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_user_prompt_includes_options():
    m = _load()
    z = {"tresc": "Treść.", "typ": "zamkniete", "opcje": {"A": "1", "B": "2"}, "opis_obrazka": ""}
    p = m.build_user_prompt(z)
    assert "Opcje odpowiedzi:" in p and "A. 1" in p


def test_make_example_shape():
    m = _load()
    z = {
        "id": 1, "typ": "zamkniete", "punkty_max": 1, "tresc": "Treść.",
        "opcje": {"A": "1", "B": "2"}, "opis_obrazka": "",
        "odpowiedz": "A", "rozwiazanie": "Bo 1 < 2.",
    }
    ex = m.make_example(z, sys_zamk="SYS-Z", sys_otw="SYS-O")
    roles = [msg["role"] for msg in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][0]["content"] == "SYS-Z"
    assert ex["messages"][2]["content"].endswith("<odpowiedz>A</odpowiedz>")
    assert "Bo 1 < 2." in ex["messages"][2]["content"]


def test_make_example_uses_open_system_prompt():
    m = _load()
    z = {"id": 15, "typ": "otwarte", "punkty_max": 2, "tresc": "Oblicz.",
         "opcje": {}, "opis_obrazka": "", "odpowiedz": "42", "rozwiazanie": "Liczymy."}
    ex = m.make_example(z, sys_zamk="SYS-Z", sys_otw="SYS-O")
    assert ex["messages"][0]["content"] == "SYS-O"
