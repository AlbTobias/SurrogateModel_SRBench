import hashlib
import json

from scripts.analyze_benchmark_results import sidecar_matches_result


def test_sidecar_must_match_current_result_hash(tmp_path) -> None:
    result = tmp_path / "seed-2.json"
    sidecar = tmp_path / "seed-2.analysis.json"
    result.write_text('{"value": 1}\n', encoding="utf-8")
    sidecar.write_text(
        json.dumps(
            {"source_result_sha256": hashlib.sha256(result.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )

    assert sidecar_matches_result(result, sidecar)
    result.write_text('{"value": 2}\n', encoding="utf-8")
    assert not sidecar_matches_result(result, sidecar)


def test_missing_or_invalid_sidecar_is_not_current(tmp_path) -> None:
    result = tmp_path / "seed-2.json"
    sidecar = tmp_path / "seed-2.analysis.json"
    result.write_text("{}\n", encoding="utf-8")

    assert not sidecar_matches_result(result, sidecar)
    sidecar.write_text("not JSON", encoding="utf-8")
    assert not sidecar_matches_result(result, sidecar)
