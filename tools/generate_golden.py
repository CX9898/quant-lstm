#!/usr/bin/env python3
"""校验权威 Golden JSON，并机械生成统一的 C++ fixture。"""

import argparse
import json
import math
import pathlib
import struct
import sys
from typing import Any

try:
    from strict_jsonschema import StrictDraft202012Validator
except ModuleNotFoundError:
    from tools.strict_jsonschema import StrictDraft202012Validator


_KINDS = {"primitive", "cell", "recurrent"}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def load_json(path: pathlib.Path) -> dict[str, Any]:
    def reject_non_finite(token: str) -> None:
        raise ValueError(f"{path} 包含非有限 JSON number: {token}")

    document = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_non_finite,
    )
    if not isinstance(document, dict):
        raise ValueError(f"{path} 的根必须是对象")
    return document


def _float32_bits(value: float) -> bytes:
    try:
        return struct.pack(">f", value)
    except OverflowError as error:
        raise ValueError("float32 字符串超出有限范围") from error


def canonical_float32(text: str) -> str:
    """返回与 C++ to_chars(general) 一致的最短 float32 十进制表示。"""
    try:
        parsed = float(text)
    except ValueError as error:
        raise ValueError(f"非法 float32 字符串: {text}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"float32 字符串必须表示有限数: {text}")

    bits = _float32_bits(parsed)
    value = struct.unpack(">f", bits)[0]
    if value.is_integer():
        integer_candidate = str(int(value))
        if _float32_bits(float(integer_candidate)) == bits:
            return integer_candidate
    for precision in range(1, 10):
        candidate = format(value, f".{precision}g")
        if _float32_bits(float(candidate)) == bits:
            return candidate
    raise AssertionError("无法格式化可往返的 float32")


def validate_tensor_contracts(document: dict[str, Any]) -> None:
    case_id = document["case_id"]

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if set(value) == {"dtype", "shape", "data"}:
                expected_size = math.prod(value["shape"])
                if len(value["data"]) != expected_size:
                    raise ValueError(
                        f"{case_id} tensor {path} shape/data 数量不一致"
                    )
                if value["dtype"] == "float32":
                    for item in value["data"]:
                        if canonical_float32(item) != item:
                            raise ValueError(
                                f"{case_id} tensor {path} 含非 canonical "
                                f"float32 字符串: {item}"
                            )
                return
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(document, "")


def _kind_from_path(spec_directory: pathlib.Path, path: pathlib.Path) -> str:
    relative = path.relative_to(spec_directory)
    if spec_directory.name in _KINDS:
        if len(relative.parts) != 1:
            raise ValueError(f"Golden 文件必须直接位于 kind 目录: {path}")
        return spec_directory.name
    if len(relative.parts) != 2 or relative.parts[0] not in _KINDS:
        raise ValueError(
            f"Golden 文件必须位于 spec/{{primitive,cell,recurrent}}/*.json: {path}"
        )
    return relative.parts[0]


def load_documents(
    spec_directory: pathlib.Path, schema_path: pathlib.Path
) -> list[dict[str, Any]]:
    schema = load_json(schema_path)
    StrictDraft202012Validator.check_schema(schema)
    validator = StrictDraft202012Validator(schema)
    documents: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for path in sorted(spec_directory.rglob("*.json")):
        document = load_json(path)
        validator.validate(document)
        expected_kind = _kind_from_path(spec_directory, path)
        if document["kind"] != expected_kind:
            raise ValueError(
                f"{path} 的 kind={document['kind']} 与目录 {expected_kind} 不一致"
            )
        validate_tensor_contracts(document)
        case_id = document["case_id"]
        if case_id in seen_case_ids:
            raise ValueError(f"重复 Golden case_id: {case_id}")
        seen_case_ids.add(case_id)
        documents.append(document)

    if not documents:
        raise ValueError("未找到 Golden JSON 文件")
    return documents


def render(spec_directory: pathlib.Path, schema_path: pathlib.Path) -> str:
    documents = load_documents(spec_directory, schema_path)
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <string_view>",
        "",
        "namespace quant_lstm::test {",
        "",
        "struct GoldenDocument {",
        "    std::string_view case_id;",
        "    std::string_view kind;",
        "    std::string_view execution_model;",
        "    std::string_view json;",
        "};",
        "",
        (
            "inline constexpr std::array<GoldenDocument, "
            f"{len(documents)}> kGoldenDocuments{{{{"
        ),
    ]
    for document in documents:
        canonical = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if ')golden"' in canonical:
            raise ValueError(
                f"{document['case_id']} 无法安全写入 C++ raw string fixture"
            )
        lines.append(
            '    {"'
            + document["case_id"]
            + '", "'
            + document["kind"]
            + '", "'
            + document["execution_model"]
            + '", R"golden('
            + canonical
            + ')golden"},'
        )
    lines.extend(["}};", "", "}  // namespace quant_lstm::test", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    directory_group = parser.add_mutually_exclusive_group(required=True)
    directory_group.add_argument("--spec-dir", type=pathlib.Path)
    directory_group.add_argument("--golden-dir", dest="spec_dir", type=pathlib.Path)
    parser.add_argument("--schema", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    generated = render(args.spec_dir, args.schema)
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != generated
        ):
            print(f"生成的 fixture 已过期: {args.output}", file=sys.stderr)
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
