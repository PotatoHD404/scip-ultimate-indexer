from __future__ import annotations

from ultimate_indexer.ranking_rules import (
    clean_symbol_display_name,
    is_generated_path,
    is_rankable_symbol,
    is_test_path,
)


def test_is_test_path_detects_test_code() -> None:
    assert is_test_path("tests/test_query.py")
    assert is_test_path("pkg/__tests__/router.test.ts")
    assert is_test_path("internal/auth/auth_test.go")
    assert is_test_path("spec/models/user.spec.ts")
    assert is_test_path("tests/conftest.py")
    assert is_test_path("app/testdata/sample.py")
    assert not is_test_path("src/ultimate_indexer/query.py")
    assert not is_test_path("services/contest/runner.py")  # 'test' inside a word
    assert not is_test_path("src/latest_results.py")


def test_is_generated_path_detects_generated_outputs() -> None:
    assert is_generated_path("services/app/.next/types/routes.d.ts")
    assert is_generated_path("services/app/proto/zitadel/object.ts")
    assert is_generated_path("apps/billing/payments-connector/proto/gen/go/payment.pb.go")
    assert is_generated_path("sdk/python/generated/payment_pb2.pyi")
    assert not is_generated_path("services/app/lib/router.ts")


def test_is_rankable_symbol_rejects_unknown_module_and_generated() -> None:
    assert not is_rankable_symbol("services/app/lib/router.ts", "Unknown")
    assert not is_rankable_symbol("services/app/lib/router.ts", "Module")
    assert not is_rankable_symbol("services/app/proto/zitadel/object.ts", "Interface")
    assert is_rankable_symbol("services/app/lib/router.ts", "Function")


def test_clean_symbol_display_name_prefers_docstring_and_symbol_tail() -> None:
    assert clean_symbol_display_name(
        symbol="scip-typescript npm demo 1.0 lib/`router.ts`/ROUTING.",
        display_name="",
        docstring="```ts\nconst ROUTING: Record<string, string>\n```",
        relative_path="lib/router.ts",
    ) == "ROUTING"
    assert clean_symbol_display_name(
        symbol="scip-typescript npm demo 1.0 lib/`i18n.ts`/getDictionary().",
        display_name="",
        docstring="",
        relative_path="lib/i18n.ts",
    ) == "getDictionary"


def test_clean_symbol_display_name_extracts_member_leafs_from_scip_descriptor() -> None:
    assert clean_symbol_display_name(
        symbol="scip-go gomod demo `models`/Table#Validate().",
        display_name="",
        docstring="",
        relative_path="models/table.go",
    ) == "Validate"
    assert clean_symbol_display_name(
        symbol="scip-go gomod demo `models`/Table#Name:",
        display_name="",
        docstring="",
        relative_path="models/table.go",
    ) == "Name"
