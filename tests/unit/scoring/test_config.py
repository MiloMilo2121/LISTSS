from __future__ import annotations

from pathlib import Path

import pytest

from list_engine.scoring.config import (
    SegmentConfigError,
    load_segment_config,
    load_segment_configs,
)

SEGMENT_DIRECTORY = Path("configs/segments")


def test_example_segments_load_as_two_versioned_hypotheses() -> None:
    configs = load_segment_configs(SEGMENT_DIRECTORY)

    assert {config.scoring_version for config in configs} == {
        "construction_north_it@v1",
        "manufacturing_north_it@v1",
    }
    assert all(config.status == "hypothesis" for config in configs)
    assert all(config.signal_lookback_days == 90 for config in configs)
    assert all(config.capacity.t1_per_bdr == 35 for config in configs)
    assert all(config.capacity.t2_per_bdr == 70 for config in configs)


def test_example_segments_preserve_the_exact_40_40_20_rubric() -> None:
    for config in load_segment_configs(SEGMENT_DIRECTORY):
        assert config.weights.fit.model_dump() == {
            "ateco": 10,
            "revenue": 10,
            "structure": 10,
            "digital_maturity": 5,
            "geography": 5,
        }
        assert config.weights.signal.model_dump() == {
            "fresh_trigger": 15,
            "growth": 10,
            "administrative_hiring": 10,
            "engagement": 5,
        }
        assert config.weights.reachability.model_dump() == {
            "decision_maker": 8,
            "phone": 6,
            "email": 3,
            "multi_source": 3,
        }


@pytest.mark.parametrize("document", ["- not\n- an\n- object\n", "null\n", "a scalar\n"])
def test_loader_rejects_non_object_documents(tmp_path: Path, document: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(SegmentConfigError, match="must contain a YAML object"):
        load_segment_config(path)


@pytest.mark.parametrize(
    ("existing", "replacement", "error"),
    [
        ("schema_version: 1", "schema_version: 1\nunexpected: value", "Extra inputs"),
        ("signal_lookback_days: 90", 'signal_lookback_days: "90"', "valid integer"),
        ("schema_version: 1", "schema_version: true", "non-integer schema_version"),
    ],
)
def test_loader_rejects_unknown_and_coercive_values(
    tmp_path: Path,
    existing: str,
    replacement: str,
    error: str,
) -> None:
    source = SEGMENT_DIRECTORY / "manufacturing_north_it.v1.yaml"
    document = source.read_text(encoding="utf-8")
    document = document.replace(existing, replacement, 1)
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(SegmentConfigError, match=error):
        load_segment_config(path)


def test_directory_loader_rejects_duplicate_segment_versions(tmp_path: Path) -> None:
    source = SEGMENT_DIRECTORY / "manufacturing_north_it.v1.yaml"
    document = source.read_text(encoding="utf-8")
    (tmp_path / "first.yaml").write_text(document, encoding="utf-8")
    (tmp_path / "second.yml").write_text(document, encoding="utf-8")

    with pytest.raises(SegmentConfigError, match="duplicate segment configuration version"):
        load_segment_configs(tmp_path)


def test_directory_loader_rejects_config_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.yaml"
    source = SEGMENT_DIRECTORY / "manufacturing_north_it.v1.yaml"
    outside.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "escaped.yaml").symlink_to(outside)

    with pytest.raises(SegmentConfigError, match="symlinks are not allowed"):
        load_segment_configs(tmp_path)
