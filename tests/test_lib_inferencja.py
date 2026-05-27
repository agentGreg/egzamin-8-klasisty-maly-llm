from scripts.lib_inferencja import build_user_prompt, extract_odpowiedz


def test_user_prompt_closed_has_options():
    z = {"tresc": "T.", "typ": "zamkniete", "opcje": {"A": "1", "B": "2"}, "opis_obrazka": ""}
    p = build_user_prompt(z)
    assert "Opcje odpowiedzi:" in p and "A. 1" in p


def test_user_prompt_open_has_figure_desc():
    z = {"tresc": "Oblicz.", "typ": "otwarte", "opcje": {}, "opis_obrazka": "trójkąt ABC"}
    p = build_user_prompt(z)
    assert "Opis rysunku" in p and "trójkąt ABC" in p


def test_extract_tag():
    assert extract_odpowiedz("blah <odpowiedz>BD</odpowiedz>") == "BD"


def test_extract_tag_absent():
    assert extract_odpowiedz("no tag here") == ""
