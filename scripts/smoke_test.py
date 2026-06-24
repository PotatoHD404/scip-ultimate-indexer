#!/usr/bin/env python3
"""End-to-end smoke test for scip-ultimate-indexer.

Builds a throwaway Python project in a temp dir, drives the real CLI
(`python -m ultimate_indexer`) through index / top-symbols / query / tree, and
asserts the output is sane. Exits non-zero on the first failure so it is usable
as a CI gate or a quick local sanity check.

Usage:
    python scripts/smoke_test.py            # uses the deterministic hash backend
    SMOKE_BACKEND=auto python scripts/smoke_test.py

What it guards (regressions this would have caught):
  * `index` must not crash with no embedding runtime installed (auto -> hash).
  * The built-in zero-config Python SCIP emitter must produce real
    Function/Class/Method symbols (not just docs / file sections).
  * `top-symbols` must rank actual code symbols, not be dominated by a doc.
  * `query` must return the lexically relevant code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FILES: dict[str, str] = {
    "pkg/__init__.py": "",
    "pkg/models.py": (
        '"""Domain models."""\n\n'
        "class User:\n"
        '    """A user of the system."""\n\n'
        "    def __init__(self, name: str, email: str) -> None:\n"
        "        self.name = name\n"
        "        self.email = email\n\n"
        "    def as_dict(self) -> dict:\n"
        '        return {"name": self.name, "email": self.email}\n'
    ),
    "pkg/services.py": (
        '"""Service layer."""\n\n'
        "from .models import User\n\n\n"
        "class GreetingService:\n"
        '    """Builds greetings for users."""\n\n'
        "    def build_greeting(self, user: User, excited: bool = False) -> str:\n"
        '        greeting = f"Hello, {user.name}"\n'
        "        if excited:\n"
        '            greeting = greeting + "!"\n'
        "        return greeting\n\n\n"
        "def serialize_user(user: User) -> dict:\n"
        '    """Serialize a user to a dict payload."""\n'
        "    return user.as_dict()\n"
    ),
    "app.py": (
        '"""Entry point."""\n\n'
        "from pkg.models import User\n"
        "from pkg.services import GreetingService\n\n\n"
        "def run() -> str:\n"
        '    """Greet a demo user."""\n'
        '    user = User(name="Ada", email="ada@example.com")\n'
        "    return GreetingService().build_greeting(user, excited=True)\n"
    ),
    "docs/guide.md": (
        "# Authentication Guide\n\n"
        "Authentication joins users to tenants by tenant id.\n\n"
        "## Tokens\n\n"
        "Tokens are signed and short lived.\n"
    ),
}

PASSED = 0
FAILED = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}")
    if ok:
        PASSED += 1
    else:
        FAILED += 1
        if detail:
            print("       " + detail.replace("\n", "\n       "))


def run_cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ultimate_indexer", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def main() -> int:
    backend = os.environ.get("SMOKE_BACKEND", "hash")
    base_env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        # Deterministic + fast: external SCIP tools (scip-python = full pyright
        # run) vary by host; the built-in emitter is what these checks assert.
        "ULTIMATE_INDEXER_DISABLE_EXTERNAL_SCIP": "1",
    }
    # The deterministic env used for functional assertions.
    env = {**base_env, "ULTIMATE_INDEXER_EMBEDDING_BACKEND": backend}

    with tempfile.TemporaryDirectory(prefix="sui-smoke-") as tmp:
        project = Path(tmp) / "project"
        for rel, content in FILES.items():
            target = project / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        # 0. Zero-config regression: `index` with the auto backend and NO
        #    ULTIMATE_INDEXER_EMBEDDING_BACKEND env must not crash even when the
        #    llama-cpp runtime is unavailable (it must degrade to hash).
        zero_env = {k: v for k, v in base_env.items()
                    if k != "ULTIMATE_INDEXER_EMBEDDING_BACKEND"}
        zc_project = Path(tmp) / "zeroconf"
        for rel, content in FILES.items():
            target = zc_project / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        zc = run_cli(["index", str(zc_project), "--no-progress",
                      "--embedding-backend", "auto"], zero_env)
        check("zero-config `index --embedding-backend auto` exits 0 (no llama_cpp crash)",
              zc.returncode == 0, zc.stderr[-1500:])

        # 1. Index the deterministic project.
        idx = run_cli(["index", str(project), "--no-progress"], env)
        check("`index` exits 0", idx.returncode == 0, idx.stderr[-1500:])
        out = idx.stdout
        # The summary table reports a Symbols column; just require it ran and
        # produced symbols (the table renders numbers in a row).
        check("`index` reports symbols", "Symbols" in out and idx.returncode == 0,
              out[-800:])

        # 2. top-symbols must surface real CODE symbols, not only docs/sections.
        top = run_cli(["top-symbols", str(project), "--limit", "10"], env)
        tout = top.stdout
        check("`top-symbols` exits 0", top.returncode == 0, top.stderr[-1500:])
        has_code_kind = any(k in tout for k in ("[Class]", "[Function]", "[Method]"))
        check("`top-symbols` ranks real code symbols (Class/Function/Method)",
              has_code_kind, tout[-800:])
        names_present = sum(n in tout for n in ("User", "GreetingService",
                                                "build_greeting", "serialize_user", "run"))
        check("`top-symbols` includes ≥2 known code symbol names",
              names_present >= 2, tout[-800:])

        # 3. query must return the lexically relevant code.
        q = run_cli(["query", str(project), "build greeting service", "--limit", "5"], env)
        qout = q.stdout
        check("`query` exits 0", q.returncode == 0, q.stderr[-1500:])
        check("`query 'build greeting service'` returns services.py / build_greeting",
              ("services.py" in qout or "build_greeting" in qout), qout[-800:])

        q2 = run_cli(["query", str(project), "serialize user payload", "--limit", "5"], env)
        check("`query 'serialize user payload'` returns serialize_user",
              q2.returncode == 0 and ("serialize_user" in q2.stdout or "services.py" in q2.stdout),
              q2.stdout[-800:])

        # 4. tree must render and include the code package.
        tree = run_cli(["tree", str(project)], env)
        check("`tree` exits 0", tree.returncode == 0, tree.stderr[-1500:])
        check("`tree` shows the code package (pkg/)", "pkg" in tree.stdout, tree.stdout[-800:])

        # 5. Tier-1 search upgrades (all default-on).
        # Reranker: an exact symbol-name query surfaces that symbol.
        rr = run_cli(["query", str(project), "build_greeting", "--scope", "code", "--limit", "5"], env)
        check("reranker: `query build_greeting` surfaces build_greeting",
              rr.returncode == 0 and "build_greeting" in rr.stdout, rr.stdout[-800:])

        # Vocabulary expansion: the abbreviation 'svc' is never written in the
        # source, yet matches GreetingService via the FTS-only expansion field.
        exp = run_cli(["query", str(project), "svc", "--scope", "code", "--limit", "5"], env)
        check("vocab expansion: `query svc` matches GreetingService",
              exp.returncode == 0 and "GreetingService" in exp.stdout, exp.stdout[-800:])

        # HyDE: a natural-language question still returns code results.
        hq = run_cli(["query", str(project), "how is a user greeted", "--scope", "code", "--limit", "5"], env)
        check("HyDE: natural-language query returns code",
              hq.returncode == 0 and bool(hq.stdout.strip()), hq.stderr[-800:])

        # Context-personalized ranking: the --focus flag is accepted and works.
        foc = run_cli(["query", str(project), "user", "--scope", "code", "--limit", "5",
                       "--focus", "app.py"], env)
        check("personalized ranking: `query --focus app.py` exits 0 with results",
              foc.returncode == 0 and bool(foc.stdout.strip()), foc.stderr[-800:])

    print()
    print(f"smoke: {PASSED} passed, {FAILED} failed (backend={backend})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
