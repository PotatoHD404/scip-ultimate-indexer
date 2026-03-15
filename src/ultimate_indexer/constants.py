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
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    "__pycache__",
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
    "vendor",
]

DEFAULT_IGNORED_DIR_NAMES = {
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    "target",
    "out",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "__pycache__",
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
    ".py": ("python", "scip-python", "pip install scip-python"),
    ".pyw": ("python", "scip-python", "pip install scip-python"),
    ".pyi": ("python", "scip-python", "pip install scip-python"),
    ".ipynb": ("python", "scip-python", "pip install scip-python"),
    ".ts": ("typescript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".tsx": ("typescript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".js": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".jsx": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".mjs": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".cjs": ("javascript", "scip-typescript", "npm install -g @sourcegraph/scip-typescript"),
    ".go": ("go", "scip-go", "go install github.com/sourcegraph/scip-go/...@latest"),
    ".rs": ("rust", "scip-rust", "cargo install scip-rust"),
    ".java": ("java", "scip-java", "see https://github.com/sourcegraph/scip-java"),
    ".cpp": ("cpp", "scip-clang", "brew install llvm"),
    ".hpp": ("cpp", "scip-clang", "brew install llvm"),
    ".cc": ("cpp", "scip-clang", "brew install llvm"),
    ".hh": ("cpp", "scip-clang", "brew install llvm"),
    ".cxx": ("cpp", "scip-clang", "brew install llvm"),
    ".c": ("c", "scip-clang", "brew install llvm"),
    ".h": ("cpp", "scip-clang", "brew install llvm"),
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
