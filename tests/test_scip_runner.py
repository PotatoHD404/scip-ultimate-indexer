from __future__ import annotations

from pathlib import Path

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
    monkeypatch.setattr("ultimate_indexer.scip_runner.shutil.which", lambda _: None)
    files = [tmp_path / "main.ts"]

    report = run_scip_indexers(tmp_path, files, tmp_path)

    assert report.results == []
    assert not report.failed
    assert len(report.missing) == 1
    assert report.missing[0].language == "typescript"
    assert report.missing[0].binary_name == "scip-typescript"
