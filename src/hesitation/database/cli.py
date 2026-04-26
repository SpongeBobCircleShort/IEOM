from __future__ import annotations

import argparse
import json

from hesitation.database.pipeline import (
    build_splits,
    derive_labels_and_audit,
    export_for_models,
    normalize_chico,
    run_benchmark_export,
    run_qc_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real dataset onboarding CLI (CHICO-first)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("normalize-chico")
    n.add_argument("--raw", required=True)
    n.add_argument("--mapping", default="merged_database/configs/chico_mapping_rules.yaml")
    n.add_argument("--output", required=True)
    n.add_argument("--report", required=True)

    d = sub.add_parser("derive-labels")
    d.add_argument("--input", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--audit", required=True)
    d.add_argument("--horizon-frames", type=int, default=15)

    q = sub.add_parser("run-qc")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--dataset-name", default="chico")

    s = sub.add_parser("build-splits")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)

    e = sub.add_parser("export-model-input")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)

    b = sub.add_parser("run-benchmark")
    b.add_argument("--input", required=True)
    b.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "normalize-chico":
        out = normalize_chico(args.raw, args.mapping, args.output, args.report)
    elif args.cmd == "derive-labels":
        out = derive_labels_and_audit(args.input, args.output, args.audit, args.horizon_frames)
    elif args.cmd == "run-qc":
        out = run_qc_report(args.input, args.output, args.dataset_name)
    elif args.cmd == "build-splits":
        out = build_splits(args.input, args.output)
    elif args.cmd == "export-model-input":
        out = export_for_models(args.input, args.output)
    elif args.cmd == "run-benchmark":
        out = run_benchmark_export(args.input, args.output_dir)
    else:
        raise ValueError(args.cmd)

    print(json.dumps({"result": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
