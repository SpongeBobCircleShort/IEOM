import json
from pathlib import Path

from hesitation.database.chico_adapter import CHICOAdapter
from hesitation.database.mapping import load_chico_mapping_pack


RAW = "tests/fixtures/chico/raw/chico_realistic_sample.jsonl"
MAPPING = "merged_database/configs/chico_mapping_rules.yaml"


def test_chico_adapter_parses_realistic_fixture() -> None:
    pack = load_chico_mapping_pack(MAPPING)
    adapter = CHICOAdapter(pack)
    records, report = adapter.normalize(RAW)
    assert len(records) == 112
    assert report.unsupported_fields.get("extra_sensor", 0) > 0


def test_mapping_report_regression_matches_golden() -> None:
    pack = load_chico_mapping_pack(MAPPING)
    adapter = CHICOAdapter(pack)
    records, report = adapter.normalize(RAW)
    golden = json.loads(Path("tests/fixtures/chico/golden/expected_mapping_report.json").read_text(encoding="utf-8"))
    assert report.dataset_name == golden["dataset_name"]
    assert report.records_total == golden["records_total"]
    assert report.mapped_records == len(records)
    assert golden["unsupported_field_expected"] in report.unsupported_fields
