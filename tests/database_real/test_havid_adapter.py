import json
from pathlib import Path

from hesitation.database.havid_adapter import HAVIDAdapter
from hesitation.database.mapping import load_havid_mapping_pack


RAW = "tests/fixtures/havid/raw/havid_realistic_sample.jsonl"
MAPPING = "merged_database/configs/havid_mapping_rules.yaml"


def test_havid_adapter_parses_realistic_fixture() -> None:
    pack = load_havid_mapping_pack(MAPPING)
    adapter = HAVIDAdapter(pack)
    records, report = adapter.normalize(RAW)
    assert len(records) == 104
    assert report.unsupported_fields.get("sensor_vendor", 0) > 0


def test_havid_mapping_regression_matches_golden() -> None:
    pack = load_havid_mapping_pack(MAPPING)
    adapter = HAVIDAdapter(pack)
    records, report = adapter.normalize(RAW)
    golden = json.loads(Path("tests/fixtures/havid/golden/expected_mapping_report.json").read_text(encoding="utf-8"))
    assert report.dataset_name == golden["dataset_name"]
    assert report.records_total == golden["records_total"]
    assert report.mapped_records == len(records)
    assert golden["unsupported_field_expected"] in report.unsupported_fields
