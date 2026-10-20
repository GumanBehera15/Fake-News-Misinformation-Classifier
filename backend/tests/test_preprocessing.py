from app.services.preprocessing import normalize_text, text_statistics

def test_normalization_is_safe_and_consistent():
    assert normalize_text("  A\tnews\nitem  ") == "A news item"
    assert text_statistics("one two")["words"] == 2
