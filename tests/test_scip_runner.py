from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ultimate_indexer.scip_runner import detect_scip_languages, run_scip_indexers


def test_detect_scip_languages_matches_codegraphcontext_surface(monkeypatch) -> None:
    monkeypatch.delenv("SCIP_LANGUAGES", raising=False)
    files = [
        Path("main.py"),
        Path("frontend.ts"),
        Path("worker.js"),
        Path("service.go"),
        Path("engine.rs"),
        Path("App.java"),
        Path("native.cpp"),
        Path("native.c"),
    ]
    detected = detect_scip_languages(files)
    assert detected == ["typescript", "go", "rust", "java", "cpp", "python"]


def test_detect_scip_languages_accepts_codegraphcontext_aliases(monkeypatch) -> None:
    monkeypatch.setenv("SCIP_LANGUAGES", "python,javascript,go,rust,java,c")
    files = [
        Path("frontend.js"),
        Path("service.go"),
        Path("engine.rs"),
        Path("App.java"),
        Path("native.c"),
        Path("main.py"),
    ]
    detected = detect_scip_languages(files)
    assert detected == ["typescript", "go", "rust", "java", "cpp", "python"]


def test_run_scip_indexers_reports_missing_tools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SCIP_LANGUAGES", raising=False)
    # This test verifies EXTERNAL tool semantics — undo the suite-wide hermetic switch.
    monkeypatch.delenv("ULTIMATE_INDEXER_DISABLE_EXTERNAL_SCIP", raising=False)
    monkeypatch.setattr("ultimate_indexer.scip_runner.shutil.which", lambda _: None)
    files = [tmp_path / "main.ts"]

    report = run_scip_indexers(tmp_path, files, tmp_path, timeout_seconds=600)

    assert report.results == []
    assert not report.failed
    assert len(report.missing) == 1
    assert report.missing[0].language == "typescript"
    assert report.missing[0].binary_name == "scip-typescript"


def test_run_scip_indexers_uses_nearest_project_roots(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    frontend = project / "frontend"
    admin = project / "admin"
    frontend.mkdir(parents=True)
    admin.mkdir(parents=True)
    (frontend / "tsconfig.json").write_text('{"compilerOptions": {}}\n', encoding="utf-8")
    (admin / "tsconfig.json").write_text('{"compilerOptions": {}}\n', encoding="utf-8")
    (frontend / "main.ts").write_text("export const main = 1\n", encoding="utf-8")
    (admin / "panel.ts").write_text("export const panel = 1\n", encoding="utf-8")

    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run(command, cwd, capture_output, text, timeout):
        calls.append((cwd, tuple(command)))
        output_path = Path(command[-1])
        output_path.write_bytes(b"scip")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.delenv("SCIP_LANGUAGES", raising=False)
    # This test verifies EXTERNAL tool semantics — undo the suite-wide hermetic switch.
    monkeypatch.delenv("ULTIMATE_INDEXER_DISABLE_EXTERNAL_SCIP", raising=False)
    monkeypatch.setattr("ultimate_indexer.scip_runner.shutil.which", lambda _: "/usr/bin/scip-typescript")
    monkeypatch.setattr("ultimate_indexer.scip_runner.subprocess.run", fake_run)

    report = run_scip_indexers(project, [frontend / "main.ts", admin / "panel.ts"], tmp_path, timeout_seconds=600)

    assert not report.missing
    assert not report.failed
    assert len(report.results) == 2
    assert {cwd for cwd, _ in calls} == {str(frontend), str(admin)}
