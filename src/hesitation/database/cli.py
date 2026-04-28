from __future__ import annotations

import argparse
import json

from hesitation.database.pipeline import (
    build_splits,
    derive_labels_and_audit,
    export_for_models,
    normalize_chico,
    normalize_havid,
    run_benchmark_export,
    run_cross_benchmark,
    run_harmonization,
    run_qc_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real dataset onboarding CLI (CHICO + HA-ViD)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    n_chico = sub.add_parser("normalize-chico")
    n_chico.add_argument("--raw", required=True)
    n_chico.add_argument("--mapping", default="merged_database/configs/chico_mapping_rules.yaml")
    n_chico.add_argument("--output", required=True)
    n_chico.add_argument("--report", required=True)

    n_havid = sub.add_parser("normalize-havid")
    n_havid.add_argument("--raw", required=True)
    n_havid.add_argument("--mapping", default="merged_database/configs/havid_mapping_rules.yaml")
    n_havid.add_argument("--output", required=True)
    n_havid.add_argument("--report", required=True)

    d = sub.add_parser("derive-labels")
    d.add_argument("--input", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--audit", required=True)
    d.add_argument("--horizon-frames", type=int, default=15)

    q = sub.add_parser("run-qc")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--dataset-name", default="dataset")

    s = sub.add_parser("build-splits")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--source-dataset", required=True)

    e = sub.add_parser("export-model-input")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)

    b = sub.add_parser("run-benchmark")
    b.add_argument("--input", required=True)
    b.add_argument("--output-dir", required=True)

    x = sub.add_parser("run-cross-benchmark")
    x.add_argument("--chico-input", required=True)
    x.add_argument("--havid-input", required=True)
    x.add_argument("--output-dir", required=True)

    h = sub.add_parser("harmonization-report")
    h.add_argument("--chico-labeled", required=True)
    h.add_argument("--havid-labeled", required=True)
    h.add_argument("--output-json", required=True)
    h.add_argument("--output-md", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "normalize-chico":
        out = normalize_chico(args.raw, args.mapping, args.output, args.report)
    elif args.cmd == "normalize-havid":
        out = normalize_havid(args.raw, args.mapping, args.output, args.report)
    elif args.cmd == "derive-labels":
        out = derive_labels_and_audit(args.input, args.output, args.audit, args.horizon_frames)
    elif args.cmd == "run-qc":
        out = run_qc_report(args.input, args.output, args.dataset_name)
    elif args.cmd == "build-splits":
        out = build_splits(args.input, args.output, args.source_dataset)
    elif args.cmd == "export-model-input":
        out = export_for_models(args.input, args.output)
    elif args.cmd == "run-benchmark":
        out = run_benchmark_export(args.input, args.output_dir)
    elif args.cmd == "run-cross-benchmark":
        out = run_cross_benchmark(args.chico_input, args.havid_input, args.output_dir)
    elif args.cmd == "harmonization-report":
        out = run_harmonization(args.chico_labeled, args.havid_labeled, args.output_json, args.output_md)
    else:
        raise ValueError(args.cmd)

    print(json.dumps({"result": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
