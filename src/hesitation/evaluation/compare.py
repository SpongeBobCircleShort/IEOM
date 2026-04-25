"""Protocol wrappers for model comparison workflows."""

from __future__ import annotations

from typing import Any

from hesitation.deep.pipeline import compare_models as compare_models_pipeline


def compare_rules_classical_deep(
    input_path: str,
    classical_model_path: str,
    deep_model_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Compare model families and write persisted report artifacts."""
    return compare_models_pipeline(
        input_path=input_path,
        classical_model_path=classical_model_path,
        deep_model_path=deep_model_path,
        output_dir=output_dir,
    )
