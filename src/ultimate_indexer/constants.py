from __future__ import annotations

import os


MAX_FILE_BYTES = round(float(os.getenv("ULTIMATE_INDEXER_MAX_FILE_SIZE_MB", "5")) * 1_000_000)
MAX_AVG_LINE_LENGTH = int(os.getenv("ULTIMATE_INDEXER_MAX_AVG_LINE_LENGTH", "500"))
MAX_CHUNK_CHARS = int(os.getenv("ULTIMATE_INDEXER_MAX_CHUNK_CHARS", "2000"))

DEFAULT_IGNORE_PATTERNS = [
    "node_modules",
    ".git",
    ".svn",
    ".hg",
    ".build",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".angular",
    "__pycache__",
    ".pycache",
    "*.pyc",
    ".venv",
    "venv",
    "env",
    ".env",
    ".tox",
    "target",
    "bin/Debug",
    "bin/Release",
    "obj",
    ".gradle",
    ".idea",
    ".vscode",
    ".vs",
    ".ultimate_indexer",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "Gemfile.lock",
    "poetry.lock",
    "*.log",
    "*.tmp",
    "*.swp",
    "*.swo",
    ".DS_Store",
    "Thumbs.db",
    "coverage",
    ".coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".nyc_output",
    ".cache",
    ".parcel-cache",
    ".pnpm-store",
    ".turbo",
    ".yarn",
    ".vercel",
    "vendor",
]

DEFAULT_IGNORED_DIR_NAMES = {
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    ".build",
    "build",
    "target",
    "out",
    ".git",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".angular",
    ".cache",
    ".parcel-cache",
    ".pnpm-store",
    ".turbo",
    ".yarn",
    ".vercel",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "__pycache__",
    ".pycache",
    ".ultimate_indexer",
}

SUPPORTED_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".py", ".pyw", ".pyi", ".ipynb",
    ".java", ".kt", ".kts", ".scala", ".sc",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".cxx",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".sh", ".bash", ".zsh",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg",
    ".md", ".mdx", ".rst", ".txt",
    ".sql",
    ".dart",
    ".lua",
    ".r", ".R",
    ".dockerfile",
    ".pl", ".pm",
    ".ex", ".exs",
    ".hs",
}

SPECIAL_FILES = {
    "Dockerfile",
    "Makefile",
    "Rakefile",
    "Gemfile",
    "Procfile",
    ".env.example",
    ".gitignore",
    ".dockerignore",
}

LANGUAGE_BY_EXTENSION = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".py": "python", ".pyw": "python", ".pyi": "python", ".ipynb": "python",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".sc": "scala",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".hh": "cpp", ".cxx": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".vue": "vue", ".svelte": "svelte",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".ini": "ini", ".cfg": "config",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".sql": "sql",
    ".dart": "dart",
    ".lua": "lua",
    ".r": "r", ".R": "r",
    ".dockerfile": "dockerfile",
    ".pl": "perl", ".pm": "perl",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell",
}

SCIP_EXTENSION_MAP = {
    ".py": ("python", "scip-python", "npm install -g @sourcegraph/scip-python"),
    ".pyw": ("python", "scip-python", "npm install -g @sourcegraph/scip-python"),
    ".pyi": ("python", "scip-python", "npm install -g @sourcegraph/scip-python"),
    ".ipynb": ("python", "scip-python", "npm install -g @sourcegraph/scip-python"),
    ".ts": ("typescript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".tsx": ("typescript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".js": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".jsx": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".mjs": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".cjs": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".go": ("go", "scip-go", "go install github.com/sourcegraph/scip-go/cmd/scip-go@latest"),
    ".rs": ("rust", "rust-analyzer", "rustup component add rust-analyzer && rustup component add rust-src"),
    ".java": ("java", "scip-java", "curl -fLo coursier https://git.io/coursier-cli && chmod +x coursier && ./coursier bootstrap --standalone -o scip-java com.sourcegraph:scip-java_2.13:0.11.2 --main com.sourcegraph.scip_java.ScipJava"),
    ".cpp": ("cpp", "scip-clang", "macOS arm64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-arm64-darwin && chmod +x ./scip-clang; Linux x86_64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-x86_64-linux && chmod +x ./scip-clang"),
    ".hpp": ("cpp", "scip-clang", "macOS arm64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-arm64-darwin && chmod +x ./scip-clang; Linux x86_64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-x86_64-linux && chmod +x ./scip-clang"),
    ".cc": ("cpp", "scip-clang", "macOS arm64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-arm64-darwin && chmod +x ./scip-clang; Linux x86_64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-x86_64-linux && chmod +x ./scip-clang"),
    ".hh": ("cpp", "scip-clang", "macOS arm64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-arm64-darwin && chmod +x ./scip-clang; Linux x86_64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-x86_64-linux && chmod +x ./scip-clang"),
    ".cxx": ("cpp", "scip-clang", "macOS arm64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-arm64-darwin && chmod +x ./scip-clang; Linux x86_64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-x86_64-linux && chmod +x ./scip-clang"),
    ".c": ("c", "scip-clang", "macOS arm64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-arm64-darwin && chmod +x ./scip-clang; Linux x86_64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-x86_64-linux && chmod +x ./scip-clang"),
    ".h": ("cpp", "scip-clang", "macOS arm64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-arm64-darwin && chmod +x ./scip-clang; Linux x86_64: curl -L -o ./scip-clang https://github.com/sourcegraph/scip-clang/releases/download/v0.4.0/scip-clang-x86_64-linux && chmod +x ./scip-clang"),
}

SCIP_RUN_ORDER = ["typescript", "javascript", "go", "rust", "java", "cpp", "c", "python"]


def parse_extra_extensions(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    return {
        item if item.startswith(".") else f".{item}"
        for item in (part.strip().lower() for part in value.split(","))
        if item
    }


EXTRA_EXTENSIONS = parse_extra_extensions(os.getenv("EXTRA_EXTENSIONS"))


def get_language_from_extension(ext: str) -> str:
    return LANGUAGE_BY_EXTENSION.get(ext, "plaintext")


def get_supported_extensions(extra_extensions: set[str] | None = None) -> set[str]:
    if not extra_extensions:
        return set(SUPPORTED_EXTENSIONS | EXTRA_EXTENSIONS)
    return set(SUPPORTED_EXTENSIONS | EXTRA_EXTENSIONS | extra_extensions)


def is_indexable_filename(filename: str, extra_extensions: set[str] | None = None) -> bool:
    if filename in SPECIAL_FILES:
        return True
    return filename.lower().endswith(tuple(get_supported_extensions(extra_extensions)))
