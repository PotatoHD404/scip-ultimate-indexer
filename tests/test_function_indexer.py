"""Tests for dual-representation function indexing."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ultimate_indexer.function_indexer import (
    FunctionBodyChunk,
    FunctionBodyChunker,
    FunctionMetadata,
    FunctionMetadataExtractor,
    chunk_function_bodies,
    extract_function_metadata,
)


class TestFunctionMetadataExtractor:
    """Test function metadata extraction from Python AST."""

    def test_simple_function(self) -> None:
        source = """
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert meta.display_name == "add"
        assert meta.kind == "Function"
        assert meta.fully_qualified_name == "test.add"
        assert meta.params == ["a", "b"]
        assert meta.param_types == {"a": "int", "b": "int"}
        assert meta.return_type == "int"
        assert meta.docstring == "Add two numbers."
        assert "async" not in meta.behavioral_tags

    def test_async_function(self) -> None:
        source = """
async def fetch_data(url: str) -> dict:
    '''Fetch data from URL.'''
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert meta.display_name == "fetch_data"
        assert "async" in meta.behavioral_tags
        assert "http" in meta.behavioral_tags

    def test_method_in_class(self) -> None:
        source = """
class UserService:
    def get_user(self, user_id: int) -> User:
        '''Get user by ID.'''
        return self.db.query(User).filter(User.id == user_id).first()
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert meta.display_name == "get_user"
        assert meta.kind == "Method"
        assert "UserService.get_user" in meta.fully_qualified_name
        assert "sql" in meta.behavioral_tags

    def test_decorators(self) -> None:
        source = """
from functools import lru_cache

@lru_cache(maxsize=128)
@retry(max_attempts=3)
def compute(x: int) -> int:
    '''Compute something.'''
    return x * 2
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert "lru_cache" in meta.decorators
        assert "retry" in meta.decorators
        assert "cache" in meta.behavioral_tags
        assert "retry" in meta.behavioral_tags

    def test_called_functions(self) -> None:
        source = """
def process_data(data: list) -> list:
    '''Process data.'''
    result = validate(data)
    transformed = transform(result)
    return save(transformed)
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert "validate" in meta.called_functions
        assert "transform" in meta.called_functions
        assert "save" in meta.called_functions

    def test_raised_exceptions(self) -> None:
        source = """
def divide(a: int, b: int) -> float:
    '''Divide a by b.'''
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert "ValueError" in meta.raised_exceptions

    def test_literals(self) -> None:
        source = """
def connect():
    '''Connect to database.'''
    url = "https://api.example.com"
    table = "users_table"
    return connect_to(table)
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert "https://api.example.com" in meta.literals

    def test_referenced_types(self) -> None:
        source = """
from typing import Optional, List

def get_items(ids: List[int]) -> Optional[Item]:
    '''Get items.'''
    return None
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        assert "List" in meta.referenced_types
        assert "Optional" in meta.referenced_types
        assert "Item" in meta.referenced_types

    def test_nested_functions(self) -> None:
        source = """
def outer(x: int) -> int:
    '''Outer function.'''
    def inner(y: int) -> int:
        '''Inner function.'''
        return y * 2
    return inner(x)
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 2
        names = [m.display_name for m in metadata_list]
        
        assert "outer" in names
        assert "inner" in names

    def test_metadata_to_index_text(self) -> None:
        source = """
def verify_token(token: str, clock: Clock) -> Claims:
    '''Verify and validate JWT claims.'''
    pass
"""
        metadata_list = extract_function_metadata("test.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        text = meta.to_index_text()
        
        assert "symbol:" in text
        assert "kind:" in text
        assert "signature:" in text
        assert "params:" in text
        assert "doc:" in text


class TestFunctionBodyChunker:
    """Test function body chunking."""

    def test_single_chunk_small_function(self) -> None:
        source = """
def add(a: int, b: int) -> int:
    return a + b
"""
        metadata_list = extract_function_metadata("test.py", source)
        chunks = chunk_function_bodies(metadata_list, source)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.chunk_index == 1
        assert chunk.total_chunks == 1
        assert chunk.chunk_type == "main"

    def test_multiple_chunks_large_function(self) -> None:
        source = """
def process_data(data: list, config: dict) -> list:
    '''Process data with configuration.'''
    # Initialize
    results = []
    
    # Validation section
    if not data:
        return []
    
    # Main processing
    for item in data:
        if validate(item):
            results.append(transform(item))
    
    # Try-catch block
    try:
        save_results(results)
    except Exception as e:
        log_error(e)
        raise
    
    return results
"""
        metadata_list = extract_function_metadata("test.py", source)
        chunks = chunk_function_bodies(metadata_list, source)
        
        # Should have at least one chunk
        assert len(chunks) >= 1
        
        # Check chunk properties
        for chunk in chunks:
            assert chunk.symbol_id == metadata_list[0].symbol_id
            assert 1 <= chunk.chunk_index <= chunk.total_chunks

    def test_chunk_body_content(self) -> None:
        source = """
def calculate(x: int) -> int:
    '''Calculate something.'''
    y = x * 2
    z = y + 1
    return z
"""
        metadata_list = extract_function_metadata("test.py", source)
        chunks = chunk_function_bodies(metadata_list, source)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Body should not include signature or docstring
        assert "def calculate" not in chunk.body
        assert "Calculate something" not in chunk.body
        assert "y = x * 2" in chunk.body or "y = x * 2" in chunk.body

    def test_chunk_to_index_text(self) -> None:
        source = """
def test_func() -> None:
    pass
"""
        metadata_list = extract_function_metadata("test.py", source)
        chunks = chunk_function_bodies(metadata_list, source)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        text = chunk.to_index_text()
        
        assert "symbol:" in text
        assert "function:" in text
        assert "chunk:" in text
        assert "type:" in text


class TestFunctionBodyChunkerLogic:
    """Test chunking logic for different code patterns."""

    def test_try_block_detection(self) -> None:
        chunker = FunctionBodyChunker(max_chunk_lines=10, min_chunk_lines=3)
        
        body_lines = [
            "result = None",
            "",
            "try:",
            "    result = risky_operation()",
            "    process(result)",
            "except ValueError as e:",
            "    log_error(e)",
            "    raise",
            "",
            "return result",
        ]
        
        split_points = chunker._find_split_points(body_lines)
        
        # Should detect try block as split point
        assert len(split_points) >= 1

    def test_loop_detection(self) -> None:
        chunker = FunctionBodyChunker(max_chunk_lines=10, min_chunk_lines=3)
        
        body_lines = [
            "results = []",
            "",
            "for item in items:",
            "    if validate(item):",
            "        results.append(transform(item))",
            "",
            "return results",
        ]
        
        split_points = chunker._find_split_points(body_lines)
        
        # Should have at least one chunk
        assert len(split_points) >= 1

    def test_section_comment_detection(self) -> None:
        chunker = FunctionBodyChunker(max_chunk_lines=10, min_chunk_lines=3)
        
        body_lines = [
            "# ========== Initialization ==========",
            "config = load_config()",
            "",
            "# ========== Processing ==========",
            "for item in items:",
            "    process(item)",
        ]
        
        split_points = chunker._find_split_points(body_lines)
        
        # Should detect section comment as split point
        assert len(split_points) >= 1


class TestDualDocumentIndexing:
    """Integration tests for dual-document indexing."""

    def test_metadata_and_body_documents_created(self) -> None:
        """Test that both metadata and body documents are created for functions."""
        source = """
def authenticate_user(username: str, password: str) -> User:
    '''Authenticate user with credentials.'''
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise UserNotFoundError()
    if not verify_password(password, user.hashed_password):
        raise InvalidPasswordError()
    return user
"""
        metadata_list = extract_function_metadata("auth.py", source)
        chunks = chunk_function_bodies(metadata_list, source)
        
        # Should have metadata
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        # Metadata should contain expected fields
        assert meta.fully_qualified_name == "auth.authenticate_user"
        assert meta.docstring == "Authenticate user with credentials."
        assert "auth" in meta.behavioral_tags
        assert "validation" in meta.behavioral_tags
        
        # Should have body chunks
        assert len(chunks) >= 1

    def test_metadata_contains_structured_data(self) -> None:
        """Test that metadata contains structured data for filtering."""
        source = """
@lru_cache(maxsize=100)
def get_config(key: str, default: str = "") -> str:
    '''Get configuration value.'''
    return CONFIG.get(key, default)
"""
        metadata_list = extract_function_metadata("config.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        # Check structured data
        assert "lru_cache" in meta.decorators
        assert "key" in meta.params
        assert "default" in meta.params
        assert meta.return_type == "str"
        assert "cache" in meta.behavioral_tags

    def test_body_chunk_preserves_symbol_identity(self) -> None:
        """Test that body chunks preserve tie-back to symbol."""
        source = """
def complex_calculation(x: int, y: int) -> int:
    '''Perform complex calculation.'''
    # Step 1: Validate inputs
    if x < 0 or y < 0:
        raise ValueError("Inputs must be positive")
    
    # Step 2: Compute intermediate values
    a = x * 2
    b = y * 3
    
    # Step 3: Combine results
    return a + b
"""
        metadata_list = extract_function_metadata("calc.py", source)
        chunks = chunk_function_bodies(metadata_list, source)
        
        assert len(chunks) >= 1
        
        # Each chunk should tie back to the symbol
        for chunk in chunks:
            assert chunk.symbol_id == metadata_list[0].symbol_id
            assert "complex_calculation" in chunk.signature

    def test_index_text_format(self) -> None:
        """Test that index text format is correct for embedding."""
        source = """
def api_call(endpoint: str) -> dict:
    '''Make API call.'''
    import requests
    response = requests.get(f"https://api.example.com/{endpoint}")
    return response.json()
"""
        metadata_list = extract_function_metadata("api.py", source)
        
        assert len(metadata_list) == 1
        meta = metadata_list[0]
        
        text = meta.to_index_text()
        
        # Should have expected format
        lines = text.split("\n")
        assert any(line.startswith("symbol:") for line in lines)
        assert any(line.startswith("kind:") for line in lines)
        assert any(line.startswith("signature:") for line in lines)
        assert any(line.startswith("params:") for line in lines)
        assert any(line.startswith("calls:") for line in lines)
        assert any(line.startswith("tags:") for line in lines)
        assert any(line.startswith("literals:") for line in lines)