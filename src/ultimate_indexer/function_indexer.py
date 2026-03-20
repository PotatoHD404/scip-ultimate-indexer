"""Function indexing with dual representation: metadata + body chunks.

This module implements code search indexing with two documents per function:
1. A compact signature + metadata document for precise semantic retrieval
2. One or more separate body documents for implementation-level recall

The metadata document contains:
- Fully qualified name
- Raw signature
- Normalized signature
- Docstring or leading comment
- Parameter names
- Return type
- Decorators / annotations / modifiers
- Referenced types / classes / modules
- Called function names
- Exception types raised
- Important literals (table names, URLs, env vars, event names)
- Behavioral tags (async, retry, cache, lock, transaction, io, http, sql)

Body chunks are split logically for long functions:
- Major branches
- Try/catch blocks
- Loops with meaningful work
- Helper closures
- Sections separated by comments
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Behavioral tag patterns
ASYNC_PATTERNS = {"async", "await"}
RETRY_PATTERNS = {"retry", "retries", "backoff", "exponential"}
CACHE_PATTERNS = {"cache", "cached", "memoize", "lru_cache", "ttl"}
LOCK_PATTERNS = {"lock", "mutex", "semaphore", "acquire", "release", "with_lock"}
TRANSACTION_PATTERNS = {"transaction", "commit", "rollback", "atomic", "begin", "tx"}
IO_PATTERNS = {"read", "write", "open", "close", "file", "stream", "io"}
HTTP_PATTERNS = {"http", "https", "request", "response", "get", "post", "put", "delete", "fetch", "api", "url", "endpoint"}
SQL_PATTERNS = {"select", "insert", "update", "delete", "query", "execute", "cursor", "connection", "database", "table", "sql"}
AUTH_PATTERNS = {"auth", "token", "jwt", "oauth", "permission", "authorize", "authenticate", "credential"}
VALIDATION_PATTERNS = {"validate", "verify", "check", "assert", "ensure", "require", "sanitize"}


@dataclass(slots=True)
class FunctionMetadata:
    """Compact metadata document for a function/method."""
    
    symbol_id: str
    relative_path: str
    display_name: str
    kind: str  # Function, Method
    fully_qualified_name: str
    signature: str  # Raw signature
    normalized_signature: str  # Normalized form
    docstring: str
    params: list[str] = field(default_factory=list)
    param_types: dict[str, str] = field(default_factory=dict)
    return_type: str = ""
    decorators: list[str] = field(default_factory=list)
    referenced_types: list[str] = field(default_factory=list)
    called_functions: list[str] = field(default_factory=list)
    raised_exceptions: list[str] = field(default_factory=list)
    literals: list[str] = field(default_factory=list)
    behavioral_tags: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    
    def to_index_text(self) -> str:
        """Convert metadata to indexable text for embedding."""
        parts: list[str] = []
        
        # Core identity
        parts.append(f"symbol: {self.fully_qualified_name}")
        parts.append(f"kind: {self.kind}")
        parts.append(f"signature: {self.normalized_signature}")
        
        # Parameters
        if self.params:
            parts.append(f"params: {', '.join(self.params)}")
        
        # Return type
        if self.return_type:
            parts.append(f"returns: {self.return_type}")
        
        # Calls
        if self.called_functions:
            parts.append(f"calls: {', '.join(self.called_functions)}")
        
        # Exceptions
        if self.raised_exceptions:
            parts.append(f"raises: {', '.join(self.raised_exceptions)}")
        
        # Types
        if self.referenced_types:
            parts.append(f"types: {', '.join(self.referenced_types)}")
        
        # Tags
        if self.behavioral_tags:
            parts.append(f"tags: {', '.join(self.behavioral_tags)}")
        
        # Literals
        if self.literals:
            parts.append(f"literals: {', '.join(self.literals)}")
        
        # Docstring
        if self.docstring:
            # Use first meaningful line
            doc_line = self._first_meaningful_line(self.docstring)
            if doc_line:
                parts.append(f"doc: {doc_line}")
        
        return "\n".join(parts)
    
    def _first_meaningful_line(self, text: str) -> str:
        """Extract first meaningful line from docstring."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("```"):
                return stripped
        return ""


@dataclass(slots=True)
class FunctionBodyChunk:
    """A body chunk for a function/method."""
    
    symbol_id: str
    relative_path: str
    display_name: str
    kind: str
    signature: str
    chunk_index: int
    total_chunks: int
    body: str
    chunk_type: str  # main, branch, loop, try_block, closure, section
    start_line: int = 0
    end_line: int = 0
    
    def to_index_text(self) -> str:
        """Convert body chunk to indexable text for embedding."""
        parts: list[str] = []
        
        # Identity prefix for retrieval tie-back
        parts.append(f"symbol: {self.symbol_id}")
        parts.append(f"function: {self.display_name}")
        parts.append(f"signature: {self.signature}")
        parts.append(f"chunk: {self.chunk_index}/{self.total_chunks}")
        parts.append(f"type: {self.chunk_type}")
        parts.append("")
        parts.append(self.body)
        
        return "\n".join(parts)


class FunctionMetadataExtractor(ast.NodeVisitor):
    """Extract metadata from Python function/method AST nodes."""
    
    def __init__(self, relative_path: str, module_name: str) -> None:
        self.relative_path = relative_path
        self.module_name = module_name
        self.enclosing_context: list[str] = []
        self.metadata_list: list[FunctionMetadata] = []
    
    def _build_fqn(self, name: str) -> str:
        """Build fully qualified name."""
        parts = self.enclosing_context + [name]
        return f"{self.module_name}.{'.'.join(parts)}"
    
    def _extract_behavioral_tags(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract behavioral tags from function."""
        tags: set[str] = set()
        source = ast.unparse(node)
        source_lower = source.lower()
        
        # Check decorators
        for dec in node.decorator_list:
            dec_str = ast.unparse(dec).lower()
            if any(p in dec_str for p in CACHE_PATTERNS):
                tags.add("cache")
            if any(p in dec_str for p in RETRY_PATTERNS):
                tags.add("retry")
            if any(p in dec_str for p in LOCK_PATTERNS):
                tags.add("lock")
        
        # Check body for patterns
        if any(p in source_lower for p in ASYNC_PATTERNS):
            tags.add("async")
        if any(p in source_lower for p in RETRY_PATTERNS):
            tags.add("retry")
        if any(p in source_lower for p in CACHE_PATTERNS):
            tags.add("cache")
        if any(p in source_lower for p in LOCK_PATTERNS):
            tags.add("lock")
        if any(p in source_lower for p in TRANSACTION_PATTERNS):
            tags.add("transaction")
        if any(p in source_lower for p in IO_PATTERNS):
            tags.add("io")
        if any(p in source_lower for p in HTTP_PATTERNS):
            tags.add("http")
        if any(p in source_lower for p in SQL_PATTERNS):
            tags.add("sql")
        if any(p in source_lower for p in AUTH_PATTERNS):
            tags.add("auth")
        if any(p in source_lower for p in VALIDATION_PATTERNS):
            tags.add("validation")
        
        return sorted(tags)
    
    def _extract_literals(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract important literals from function body."""
        literals: set[str] = set()
        
        for child in ast.walk(node):
            # String literals that look like important values
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                val = child.value
                # URLs
                if val.startswith(("http://", "https://")):
                    literals.add(val)
                # Environment variable patterns
                elif re.match(r"^[A-Z][A-Z0-9_]+$", val) and len(val) > 3:
                    literals.add(val)
                # Table names (snake_case with common prefixes)
                elif re.match(r"^(?:tbl_|table_|v_|view_)?[a-z][a-z0-9_]*$", val) and len(val) > 4:
                    # Check if it looks like a table name
                    if any(kw in val.lower() for kw in ["table", "tbl", "view", "log", "config", "setting", "user", "session"]):
                        literals.add(val)
            # Numeric literals
            elif isinstance(child, ast.Constant) and isinstance(child.value, (int, float)):
                # Only include named constants or significant values
                pass  # Skip raw numbers for now
        
        return sorted(literals)
    
    def _extract_called_functions(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract called function names from function body."""
        calls: set[str] = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        
        return sorted(calls)
    
    def _extract_raised_exceptions(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract raised exception types from function body."""
        exceptions: set[str] = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Name):
                    exceptions.add(child.exc.id)
                elif isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.add(child.exc.func.id)
        
        return sorted(exceptions)
    
    def _extract_referenced_types(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract referenced type names from function signature and body."""
        types: set[str] = set()
        
        # From annotations
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation:
                self._collect_type_names(arg.annotation, types)
        if node.returns:
            self._collect_type_names(node.returns, types)
        
        # From type annotations in body
        for child in ast.walk(node):
            if isinstance(child, ast.AnnAssign) and child.annotation:
                self._collect_type_names(child.annotation, types)
        
        return sorted(types)
    
    def _collect_type_names(self, node: ast.AST, types: set[str]) -> None:
        """Recursively collect type names from annotation node."""
        if isinstance(node, ast.Name):
            types.add(node.id)
        elif isinstance(node, ast.Attribute):
            types.add(node.attr)
        elif isinstance(node, ast.Subscript):
            self._collect_type_names(node.value, types)
            if isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    self._collect_type_names(elt, types)
            else:
                self._collect_type_names(node.slice, types)
    
    def _extract_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract decorator names."""
        decorators: list[str] = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)
        return decorators
    
    def _normalize_signature(self, signature: str) -> str:
        """Normalize signature for consistent comparison."""
        # Remove extra whitespace
        sig = re.sub(r"\s+", " ", signature)
        # Remove decorator lines
        sig = "\n".join(line for line in sig.split("\n") if not line.startswith("@"))
        return sig.strip()
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition and track context."""
        self.enclosing_context.append(node.name)
        self.generic_visit(node)
        self.enclosing_context.pop()
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and extract metadata."""
        self._process_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition and extract metadata."""
        self._process_function(node)
    
    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process a function/method node and extract metadata."""
        kind = "Method" if self.enclosing_context else "Function"
        fqn = self._build_fqn(node.name)
        
        # Extract parameters
        params: list[str] = []
        param_types: dict[str, str] = {}
        
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for arg in all_args:
            params.append(arg.arg)
            if arg.annotation:
                param_types[arg.arg] = ast.unparse(arg.annotation)
        
        # Return type
        return_type = ""
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        # Signature
        signature = ast.unparse(node)
        
        # Docstring
        docstring = ast.get_docstring(node) or ""
        
        # Extract metadata
        metadata = FunctionMetadata(
            symbol_id=f"py:{self.module_name}:{'.'.join(self.enclosing_context + [node.name])}:function" if not self.enclosing_context else f"py:{self.module_name}:{'.'.join(self.enclosing_context + [node.name])}:method",
            relative_path=self.relative_path,
            display_name=node.name,
            kind=kind,
            fully_qualified_name=fqn,
            signature=signature,
            normalized_signature=self._normalize_signature(signature),
            docstring=docstring,
            params=params,
            param_types=param_types,
            return_type=return_type,
            decorators=self._extract_decorators(node),
            referenced_types=self._extract_referenced_types(node),
            called_functions=self._extract_called_functions(node),
            raised_exceptions=self._extract_raised_exceptions(node),
            literals=self._extract_literals(node),
            behavioral_tags=self._extract_behavioral_tags(node),
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        )
        
        self.metadata_list.append(metadata)
        
        # Visit children
        self.enclosing_context.append(node.name)
        self.generic_visit(node)
        self.enclosing_context.pop()


class FunctionBodyChunker:
    """Chunk function bodies into logical segments."""
    
    def __init__(
        self,
        max_chunk_lines: int = 50,
        min_chunk_lines: int = 5,
        overlap_lines: int = 3,
    ) -> None:
        self.max_chunk_lines = max_chunk_lines
        self.min_chunk_lines = min_chunk_lines
        self.overlap_lines = overlap_lines
    
    def chunk_function(
        self,
        metadata: FunctionMetadata,
        source_lines: list[str],
    ) -> list[FunctionBodyChunk]:
        """Generate body chunks for a function."""
        if metadata.start_line > len(source_lines) or metadata.end_line > len(source_lines):
            return []
        
        # Extract function body lines
        body_lines = source_lines[metadata.start_line - 1:metadata.end_line]
        
        # Remove signature line and docstring
        body_lines = self._extract_body_only(body_lines, metadata.docstring)
        
        if not body_lines:
            return []
        
        # Determine chunking strategy
        if len(body_lines) <= self.max_chunk_lines:
            # Single chunk
            return [
                FunctionBodyChunk(
                    symbol_id=metadata.symbol_id,
                    relative_path=metadata.relative_path,
                    display_name=metadata.display_name,
                    kind=metadata.kind,
                    signature=metadata.signature,
                    chunk_index=1,
                    total_chunks=1,
                    body="\n".join(body_lines),
                    chunk_type="main",
                    start_line=metadata.start_line,
                    end_line=metadata.end_line,
                )
            ]
        
        # Multiple chunks needed
        return self._chunk_body(metadata, body_lines)
    
    def _extract_body_only(self, lines: list[str], docstring: str) -> list[str]:
        """Extract only the function body, removing signature and docstring."""
        if not lines:
            return []
        
        # Skip signature (first line typically)
        start_idx = 0
        for i, line in enumerate(lines):
            if ":" in line and ("def " in line or "async def " in line):
                start_idx = i + 1
                break
        
        # Skip docstring
        if start_idx < len(lines):
            remaining = lines[start_idx:]
            stripped = [l.strip() for l in remaining]
            
            # Check for triple-quoted docstring
            if stripped and ('"""' in stripped[0] or "'''" in stripped[0]):
                quote = '"""' if '"""' in stripped[0] else "'''"
                # Single-line docstring
                if stripped[0].count(quote) >= 2:
                    start_idx += 1
                else:
                    # Multi-line docstring
                    for i, line in enumerate(stripped[1:], 1):
                        if quote in line:
                            start_idx += i + 1
                            break
                    else:
                        start_idx += 1  # Unclosed docstring
        
        # Return body lines, stripped of leading/trailing empty lines
        body = lines[start_idx:]
        while body and not body[0].strip():
            body = body[1:]
        while body and not body[-1].strip():
            body = body[:-1]
        
        return body
    
    def _chunk_body(
        self,
        metadata: FunctionMetadata,
        body_lines: list[str],
    ) -> list[FunctionBodyChunk]:
        """Split function body into logical chunks."""
        chunks: list[FunctionBodyChunk] = []
        
        # Find logical split points
        split_points = self._find_split_points(body_lines)
        
        if len(split_points) <= 1:
            # Fall back to line-based chunking
            return self._chunk_by_lines(metadata, body_lines)
        
        # Create chunks from split points
        for i, (start, end, chunk_type) in enumerate(split_points):
            chunk_lines = body_lines[start:end]
            if len(chunk_lines) < self.min_chunk_lines:
                continue
            
            chunk = FunctionBodyChunk(
                symbol_id=metadata.symbol_id,
                relative_path=metadata.relative_path,
                display_name=metadata.display_name,
                kind=metadata.kind,
                signature=metadata.signature,
                chunk_index=i + 1,
                total_chunks=len(split_points),
                body="\n".join(chunk_lines),
                chunk_type=chunk_type,
                start_line=metadata.start_line + start,
                end_line=metadata.start_line + end,
            )
            chunks.append(chunk)
        
        if not chunks:
            return self._chunk_by_lines(metadata, body_lines)
        
        return chunks
    
    def _find_split_points(
        self,
        body_lines: list[str],
    ) -> list[tuple[int, int, str]]:
        """Find logical split points in function body."""
        split_points: list[tuple[int, int, str]] = []
        
        current_start = 0
        current_type = "main"
        
        for i, line in enumerate(body_lines):
            stripped = line.strip()
            
            # Detect section comments
            if stripped.startswith("#") and "=" in stripped:
                if i > current_start + self.min_chunk_lines:
                    split_points.append((current_start, i, current_type))
                current_start = i
                current_type = "section"
            
            # Detect try blocks
            elif stripped.startswith("try:"):
                if i > current_start + self.min_chunk_lines:
                    split_points.append((current_start, i, current_type))
                current_start = i
                current_type = "try_block"
            
            # Detect except blocks
            elif stripped.startswith("except"):
                if i > current_start + self.min_chunk_lines:
                    split_points.append((current_start, i, current_type))
                current_start = i
                current_type = "except_block"
            
            # Detect for/while loops
            elif stripped.startswith(("for ", "while ")):
                # Only split if loop is significant
                loop_end = self._find_block_end(body_lines, i)
                if loop_end - i > self.min_chunk_lines and i > current_start + self.min_chunk_lines:
                    if current_start < i:
                        split_points.append((current_start, i, current_type))
                    current_start = i
                    current_type = "loop"
        
        # Add final chunk
        if current_start < len(body_lines):
            split_points.append((current_start, len(body_lines), current_type))
        
        return split_points
    
    def _find_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end of a code block starting at given index."""
        if start_idx >= len(lines):
            return start_idx + 1
        
        # Get indentation of first line
        first_line = lines[start_idx]
        base_indent = len(first_line) - len(first_line.lstrip())
        
        end_idx = start_idx + 1
        while end_idx < len(lines):
            line = lines[end_idx]
            if not line.strip():
                end_idx += 1
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent:
                break
            end_idx += 1
        
        return end_idx
    
    def _chunk_by_lines(
        self,
        metadata: FunctionMetadata,
        body_lines: list[str],
    ) -> list[FunctionBodyChunk]:
        """Chunk body by line count with overlap."""
        chunks: list[FunctionBodyChunk] = []
        
        total = len(body_lines)
        start = 0
        chunk_idx = 0
        
        while start < total:
            end = min(start + self.max_chunk_lines, total)
            
            # Add overlap (except for first chunk)
            if chunk_idx > 0 and start > 0:
                start = max(0, start - self.overlap_lines)
            
            chunk_lines = body_lines[start:end]
            if len(chunk_lines) < self.min_chunk_lines and chunk_idx > 0:
                # Merge with previous chunk
                if chunks:
                    prev = chunks[-1]
                    prev.body += "\n" + "\n".join(chunk_lines)
                    prev.total_chunks = len(chunks)
                    prev.end_line = metadata.start_line + end
                break
            
            chunk = FunctionBodyChunk(
                symbol_id=metadata.symbol_id,
                relative_path=metadata.relative_path,
                display_name=metadata.display_name,
                kind=metadata.kind,
                signature=metadata.signature,
                chunk_index=chunk_idx + 1,
                total_chunks=0,  # Will be set after all chunks created
                body="\n".join(chunk_lines),
                chunk_type="main",
                start_line=metadata.start_line + start,
                end_line=metadata.start_line + end,
            )
            chunks.append(chunk)
            
            chunk_idx += 1
            start = end
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks


def extract_function_metadata(
    relative_path: str,
    source: str,
    module_name: str | None = None,
) -> list[FunctionMetadata]:
    """Extract metadata from all functions in a Python source file."""
    try:
        tree = ast.parse(source, filename=relative_path)
    except SyntaxError:
        return []
    
    if module_name is None:
        # Derive module name from path
        path = relative_path[:-3] if relative_path.endswith(".py") else relative_path
        parts = [p for p in path.split("/") if p != "__init__"]
        module_name = ".".join(parts) or "module"
    
    extractor = FunctionMetadataExtractor(relative_path, module_name)
    extractor.visit(tree)
    
    return extractor.metadata_list


def chunk_function_bodies(
    metadata_list: list[FunctionMetadata],
    source: str,
) -> list[FunctionBodyChunk]:
    """Generate body chunks for a list of function metadata."""
    source_lines = source.splitlines()
    chunker = FunctionBodyChunker()
    
    all_chunks: list[FunctionBodyChunk] = []
    for metadata in metadata_list:
        chunks = chunker.chunk_function(metadata, source_lines)
        all_chunks.extend(chunks)
    
    return all_chunks
