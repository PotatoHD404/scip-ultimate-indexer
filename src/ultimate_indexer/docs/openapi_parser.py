"""Parser for OpenAPI 3.x specification files.

This module parses OpenAPI specifications (YAML or JSON) into structured sections,
extracting endpoints, schemas, and resolving $ref pointers. It follows patterns
similar to scip_parser.py and produces output compatible with the existing
ultimate-indexer infrastructure.

Features:
- Parse OpenAPI 3.x specs in YAML or JSON format
- Extract info section, paths/endpoints, and component schemas
- Resolve internal and external $ref pointers
- Generate rich section content with parameters, request bodies, and responses
- Create links for $ref relationships and tag groupings
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .markdown_parser import ParsedLink, SectionHeader, slugify_heading

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAPISection:
    """A parsed section from an OpenAPI specification."""
    header: SectionHeader | None
    level: int
    content: str
    start_line: int
    end_line: int
    anchor: str | None
    anchors_in_range: list[str] = field(default_factory=list)
    chunk_type: str = "openapi_endpoint"  # openapi_endpoint, openapi_schema, openapi_description
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenAPIParser:
    """Parses OpenAPI 3.x specifications into document chunks and links."""

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.raw_content = content
        self.spec: dict[str, Any] = {}
        self.links: list[ParsedLink] = []
        self.anchors: dict[str, int] = {}

    def parse(self) -> tuple[list[OpenAPISection], list[ParsedLink], dict[str, int]]:
        """Parse the OpenAPI spec into sections and links."""
        try:
            if self.file_path.endswith('.json'):
                self.spec = json.loads(self.raw_content)
            else:
                self.spec = yaml.safe_load(self.raw_content)
        except Exception as e:
            logger.error(f"Failed to parse OpenAPI spec {self.file_path}: {e}")
            return [], [], {}

        if not isinstance(self.spec, dict):
            logger.error(f"OpenAPI spec {self.file_path} is not a valid mapping")
            return [], [], {}

        sections: list[OpenAPISection] = []
        sections.extend(self._parse_info())
        sections.extend(self._parse_paths())
        sections.extend(self._parse_schemas())
        self._resolve_refs()

        return sections, self.links, self.anchors

    def _parse_info(self) -> list[OpenAPISection]:
        """Parse the info section."""
        info = self.spec.get('info', {})
        if not info:
            return []

        title = info.get('title', 'API')
        description = info.get('description', '')
        version = info.get('version', '')

        content_parts = [f"# {title}"]
        if version:
            content_parts.append(f"**Version:** {version}")
        if description:
            content_parts.append(f"\n{description}")

        anchor = slugify_heading(title)
        self.anchors[anchor] = 0

        return [OpenAPISection(
            header=SectionHeader(
                level=1, text=title, anchor=anchor,
                line_number=0, file_path=self.file_path
            ),
            level=1,
            content='\n'.join(content_parts),
            start_line=0,
            end_line=0,
            anchor=anchor,
            anchors_in_range=[anchor],
            chunk_type='openapi_description',
        )]

    def _parse_paths(self) -> list[OpenAPISection]:
        """Parse paths/endpoints into sections."""
        paths = self.spec.get('paths', {})
        sections: list[OpenAPISection] = []

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in ('get', 'post', 'put', 'patch', 'delete', 'options', 'head'):
                if method not in path_item:
                    continue
                operation = path_item[method]
                if not isinstance(operation, dict):
                    continue

                op_id = operation.get('operationId', f"{method}-{path}")
                summary = operation.get('summary', f"{method.upper()} {path}")
                description = operation.get('description', '')
                tags = operation.get('tags', [])

                # Build rich content representation
                content_parts = [
                    f"## {method.upper()} {path}",
                    f"**Operation ID:** `{op_id}`",
                ]
                if summary:
                    content_parts.append(f"**Summary:** {summary}")
                if tags:
                    content_parts.append(f"**Tags:** {', '.join(tags)}")
                if description:
                    content_parts.append(f"\n{description}")

                # Parameters
                params = operation.get('parameters', [])
                if params:
                    content_parts.append("\n### Parameters")
                    for param in params:
                        if isinstance(param, dict):
                            name = param.get('name', '?')
                            location = param.get('in', '?')
                            required = param.get('required', False)
                            desc = param.get('description', '')
                            req_marker = " *(required)*" if required else ""
                            content_parts.append(
                                f"- `{name}` ({location}){req_marker}: {desc}"
                            )

                # Request body
                req_body = operation.get('requestBody', {})
                if isinstance(req_body, dict) and req_body:
                    content_parts.append("\n### Request Body")
                    rb_desc = req_body.get('description', '')
                    if rb_desc:
                        content_parts.append(rb_desc)
                    for media_type, media_obj in req_body.get('content', {}).items():
                        schema = media_obj.get('schema', {})
                        content_parts.append(
                            f"- **{media_type}**: `{self._schema_summary(schema)}`"
                        )

                # Responses
                responses = operation.get('responses', {})
                if responses:
                    content_parts.append("\n### Responses")
                    for status, resp in responses.items():
                        if isinstance(resp, dict):
                            resp_desc = resp.get('description', '')
                            content_parts.append(f"- **{status}**: {resp_desc}")

                anchor = slugify_heading(f"{method}-{path}")
                self.anchors[anchor] = len(sections)

                sections.append(OpenAPISection(
                    header=SectionHeader(
                        level=2, text=f"{method.upper()} {path}",
                        anchor=anchor, line_number=len(sections),
                        file_path=self.file_path
                    ),
                    level=2,
                    content='\n'.join(content_parts),
                    start_line=0,
                    end_line=0,
                    anchor=anchor,
                    anchors_in_range=[anchor, slugify_heading(op_id)],
                    chunk_type='openapi_endpoint',
                    metadata={
                        'method': method,
                        'path': path,
                        'operation_id': op_id,
                        'tags': tags,
                    },
                ))

                # Tag-based links
                for tag in tags:
                    tag_anchor = slugify_heading(tag)
                    self.anchors.setdefault(tag_anchor, 0)

        return sections

    def _parse_schemas(self) -> list[OpenAPISection]:
        """Parse component schemas."""
        components = self.spec.get('components', {})
        schemas = components.get('schemas', {})
        sections: list[OpenAPISection] = []

        for name, schema in schemas.items():
            if not isinstance(schema, dict):
                continue

            content_parts = [f"## Schema: {name}"]
            desc = schema.get('description', '')
            if desc:
                content_parts.append(desc)

            schema_type = schema.get('type', 'object')
            content_parts.append(f"\n**Type:** `{schema_type}`")

            # Properties
            properties = schema.get('properties', {})
            required_fields = schema.get('required', [])
            if properties:
                content_parts.append("\n### Properties")
                for prop_name, prop_schema in properties.items():
                    if isinstance(prop_schema, dict):
                        prop_type = prop_schema.get('type', 'any')
                        prop_desc = prop_schema.get('description', '')
                        req = " *(required)*" if prop_name in required_fields else ""
                        content_parts.append(
                            f"- `{prop_name}` ({prop_type}){req}: {prop_desc}"
                        )

            anchor = slugify_heading(f"schema-{name}")
            self.anchors[anchor] = len(sections)
            # Also register a components/schemas path anchor
            ref_anchor = f"components-schemas-{slugify_heading(name)}"
            self.anchors[ref_anchor] = len(sections)

            sections.append(OpenAPISection(
                header=SectionHeader(
                    level=2, text=f"Schema: {name}",
                    anchor=anchor, line_number=len(sections),
                    file_path=self.file_path
                ),
                level=2,
                content='\n'.join(content_parts),
                start_line=0,
                end_line=0,
                anchor=anchor,
                anchors_in_range=[anchor, ref_anchor],
                chunk_type='openapi_schema',
                metadata={'schema_name': name},
            ))

        return sections

    def _resolve_refs(self):
        """Walk the spec finding $ref pointers and creating links."""
        self._walk_refs(self.spec, [])

    def _walk_refs(self, obj: Any, path: list[str]):
        """Recursively walk the spec to find $ref pointers."""
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref = obj['$ref']
                source_anchor = slugify_heading('-'.join(path[:3])) if path else None

                if ref.startswith('#/'):
                    # Internal ref
                    ref_path = ref[2:].replace('/', '-')
                    target_anchor = slugify_heading(ref_path)
                    self.links.append(ParsedLink(
                        source_file=self.file_path,
                        source_anchor=source_anchor,
                        target_raw=ref,
                        target_file=self.file_path,
                        target_anchor=target_anchor,
                        link_type='openapi_ref',
                        context_text=f"$ref: {ref}",
                    ))
                elif not ref.startswith('http'):
                    # External file ref
                    file_part = ref.split('#')[0] if '#' in ref else ref
                    anchor_part = ref.split('#')[1] if '#' in ref else None
                    if anchor_part:
                        anchor_part = slugify_heading(
                            anchor_part.lstrip('/').replace('/', '-')
                        )
                    self.links.append(ParsedLink(
                        source_file=self.file_path,
                        source_anchor=source_anchor,
                        target_raw=ref,
                        target_file=file_part,
                        target_anchor=anchor_part,
                        link_type='openapi_ref',
                        context_text=f"$ref: {ref}",
                    ))
            for key, value in obj.items():
                self._walk_refs(value, path + [key])
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._walk_refs(item, path + [str(i)])

    def _schema_summary(self, schema: dict) -> str:
        """Get a summary string for a schema."""
        if '$ref' in schema:
            return schema['$ref'].split('/')[-1]
        schema_type = schema.get('type', 'object')
        if schema_type == 'array':
            items = schema.get('items', {})
            return f"array[{self._schema_summary(items)}]"
        return schema_type
