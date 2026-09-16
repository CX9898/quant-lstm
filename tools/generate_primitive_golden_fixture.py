#!/usr/bin/env python3
"""Generate a C++ fixture header from reviewed primitive Golden JSON files."""

import argparse
import json
import math
import pathlib
import sys

from strict_jsonschema import StrictDraft202012Validator


def validate_tensor_shapes(document: dict) -> None:
    tensor_maps = [
        document["inputs"],
        document["expected"]["checkpoints"],
        document["expected"]["diagnostics"],
    ]
    for tensor_map in tensor_maps:
        for name, tensor in tensor_map.items():
            if len(tensor["data"]) != math.prod(tensor["shape"]):
                raise ValueError(
                    f"{document['case_id']} tensor {name} shape/data mismatch"
                )


def render(golden_directory: pathlib.Path, schema_path: pathlib.Path) -> str:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    StrictDraft202012Validator.check_schema(schema)
    validator = StrictDraft202012Validator(schema)
    documents: list[tuple[str, str]] = []
    seen_case_ids: set[str] = set()
    for path in sorted(golden_directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        validator.validate(document)
        validate_tensor_shapes(document)
        case_id = document["case_id"]
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate Golden case_id: {case_id}")
        seen_case_ids.add(case_id)
        canonical = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        documents.append((case_id, canonical))
    if not documents:
        raise ValueError("no primitive Golden JSON files found")

    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <string_view>",
        "",
        "namespace quant_lstm::test {",
        "",
        "struct PrimitiveGoldenDocument {",
        "    std::string_view case_id;",
        "    std::string_view json;",
        "};",
        "",
        (
            "inline constexpr std::array<PrimitiveGoldenDocument, "
            f"{len(documents)}> kPrimitiveGoldenDocuments{{{{"
        ),
    ]
    for case_id, canonical in documents:
        lines.append(f'    {{"{case_id}", R"golden({canonical})golden"}},')
    lines.extend(
        [
            "}};",
            "",
            "}  // namespace quant_lstm::test",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", type=pathlib.Path, required=True)
    parser.add_argument("--schema", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    generated = render(args.golden_dir, args.schema)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != generated:
            print(f"generated fixture is stale: {args.output}", file=sys.stderr)
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
