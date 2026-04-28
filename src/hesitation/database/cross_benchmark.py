from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hesitation.deep.pipeline import evaluate_deep, train_deep
from hesitation.ml.pipeline import evaluate_classical, train_classical


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _concat_jsonl(inputs: list[str], output: str) -> str:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as w:
        for path in inputs:
            with Path(path).open("r", encoding="utf-8") as r:
                for line in r:
                    if line.strip():
                        w.write(line)
    return str(out)


def run_cross_dataset_benchmark(
    chico_model_input: str,
    havid_model_input: str,
    output_dir: str,
    window_size: int = 10,
    horizon_frames: int = 5,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CHICO train -> HAVID test
    ct_dir = out / "chico_train"
    train_classical(chico_model_input, str(ct_dir / "classical"), window_size, 0.03, horizon_frames)
    chico_to_havid_classical = evaluate_classical(havid_model_input, str(ct_dir / "classical" / "classical_model.json"))

    train_deep(
        input_path=chico_model_input,
        output_dir=str(ct_dir / "deep"),
        window_size=window_size,
        horizon_frames=horizon_frames,
        epochs=3,
        hidden_dim=16,
        learning_rate=0.005,
        seed=11,
        batch_size=16,
    )
    chico_to_havid_deep = evaluate_deep(havid_model_input, str(ct_dir / "deep" / "deep_model.json"))

    # HAVID train -> CHICO test
    ht_dir = out / "havid_train"
    train_classical(havid_model_input, str(ht_dir / "classical"), window_size, 0.03, horizon_frames)
    havid_to_chico_classical = evaluate_classical(chico_model_input, str(ht_dir / "classical" / "classical_model.json"))

    train_deep(
        input_path=havid_model_input,
        output_dir=str(ht_dir / "deep"),
        window_size=window_size,
        horizon_frames=horizon_frames,
        epochs=3,
        hidden_dim=16,
        learning_rate=0.005,
        seed=12,
        batch_size=16,
    )
    havid_to_chico_deep = evaluate_deep(chico_model_input, str(ht_dir / "deep" / "deep_model.json"))

    # merged train/eval
    merged_input = _concat_jsonl([chico_model_input, havid_model_input], str(out / "merged_model_input.jsonl"))
    train_classical(merged_input, str(out / "merged" / "classical"), window_size, 0.03, horizon_frames)
    merged_classical = evaluate_classical(merged_input, str(out / "merged" / "classical" / "classical_model.json"))

    train_deep(
        input_path=merged_input,
        output_dir=str(out / "merged" / "deep"),
        window_size=window_size,
        horizon_frames=horizon_frames,
        epochs=3,
        hidden_dim=16,
        learning_rate=0.005,
        seed=13,
        batch_size=16,
    )
    merged_deep = evaluate_deep(merged_input, str(out / "merged" / "deep" / "deep_model.json"))

    summary = {
        "chico_train_havid_test": {
            "classical": chico_to_havid_classical,
            "deep": chico_to_havid_deep,
        },
        "havid_train_chico_test": {
            "classical": havid_to_chico_classical,
            "deep": havid_to_chico_deep,
        },
        "merged_train_eval": {
            "classical": merged_classical,
            "deep": merged_deep,
        },
        "caveats": [
            "cross-dataset transfer is sensitive to label and action normalization assumptions",
            "deep backend may fallback when torch is unavailable",
        ],
    }

    _write_json(out / "cross_dataset_benchmark_summary.json", summary)
    return summary
