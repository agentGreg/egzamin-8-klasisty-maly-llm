from scripts.metryki_format import (
    strict_compliant,
    generous_parsed,
    format_compliance,
    conditional_accuracy,
)


def test_strict_compliant_true():
    assert strict_compliant("rozumowanie <odpowiedz>C</odpowiedz>", "C") is True


def test_strict_compliant_false_when_no_tag():
    assert strict_compliant("po prostu C na końcu", "") is False


def test_strict_compliant_false_when_tag_garbage():
    assert strict_compliant("<odpowiedz>nie wiem</odpowiedz>", "") is False


def test_generous_parsed_true_via_fallback():
    assert generous_parsed("...\n#### D", "") is True


def test_format_compliance_counts_strict():
    rows = [
        {"raw": "<odpowiedz>A</odpowiedz>", "odpowiedz": "A"},
        {"raw": "luzem B na końcu B", "odpowiedz": ""},
    ]
    # 1 of 2 strict-compliant
    assert format_compliance(rows) == 0.5


def test_conditional_accuracy_ignores_unparseable():
    rows = [
        {"raw": "<odpowiedz>A</odpowiedz>", "odpowiedz": "A"},  # parsed, correct
        {"raw": "<odpowiedz>B</odpowiedz>", "odpowiedz": "B"},  # parsed, wrong
        {"raw": "zupelnie bez litery", "odpowiedz": ""},        # unparseable -> excluded
    ]
    keys = {1: "A", 2: "A", 3: "C"}
    ids = [1, 2, 3]
    # parsed: rows 1,2 -> 1 correct of 2 = 0.5
    assert conditional_accuracy(rows, ids, keys) == 0.5
