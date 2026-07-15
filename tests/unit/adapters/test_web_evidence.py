from __future__ import annotations

from list_engine.adapters.web_evidence import extract_web_evidence, normalize_italian_phone


def test_extract_web_evidence_preserves_sources_and_deduplicates() -> None:
    html = """
    <html><body>
      <footer>
        Partita IVA: IT 990.000.000-02
        <a href="mailto:INFO@AURORA.EXAMPLE">Scrivici</a>
        <a href="mailto:amministrazione@aurora.pec.it">PEC</a>
        <a href="tel:+39 02 1234 5678">Chiama</a>
      </footer>
      <script type="application/ld+json">
        {"@type":"Organization","vatID":"99000000002",
         "email":"info@aurora.example","telephone":"02 1234 5678"}
      </script>
    </body></html>
    """

    evidence = extract_web_evidence(html, source_url="https://aurora-logistica.example.invalid")

    assert [item.normalized_value for item in evidence.values_for("piva")] == ["99000000002"]
    assert [item.normalized_value for item in evidence.values_for("email")] == [
        "info@aurora.example"
    ]
    assert [item.normalized_value for item in evidence.values_for("pec")] == [
        "amministrazione@aurora.pec.it"
    ]
    assert [item.normalized_value for item in evidence.values_for("phone")] == ["+390212345678"]
    assert evidence.values_for("phone")[0].source_url.endswith(".invalid")
    assert evidence.json_ld_blocks_seen == 1


def test_malformed_or_uncited_values_are_ignored_without_throwing() -> None:
    evidence = extract_web_evidence(
        "<script type='application/ld+json'>{broken</script> P.IVA 99000000003",
        source_url="https://fixture.example.invalid",
    )

    assert evidence.values == ()
    assert evidence.json_ld_blocks_seen == 1


def test_phone_normalisation_is_conservative_and_keeps_landline_zero() -> None:
    assert normalize_italian_phone("0039 02 1234 5678") == "+390212345678"
    assert normalize_italian_phone("+39 333 123 4567") == "+393331234567"
    assert normalize_italian_phone("12345") is None
