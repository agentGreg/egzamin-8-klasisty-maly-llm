from scripts.parser_odpowiedzi import normalize, wylusk_zamkniete


def test_strict_tag_letter():
    assert wylusk_zamkniete("blah <odpowiedz>C</odpowiedz>", "C") == "C"


def test_clean_field_short_circuits():
    assert wylusk_zamkniete("irrelevant text", "BD") == "BD"


def test_boxed_fallback():
    assert wylusk_zamkniete(r"final \boxed{A}", "") == "A"


def test_gsm8k_hash_fallback():
    assert wylusk_zamkniete("rozumowanie...\n#### D", "") == "D"


def test_enumerated_last_line():
    assert wylusk_zamkniete("A. zle\nB. tez zle\nC. dobrze", "") == "C"


def test_normalize_strips_ws():
    assert normalize(" a c ") == "AC"


def test_no_match_returns_field_upper():
    assert wylusk_zamkniete("kompletnie bez litery", "") == ""
