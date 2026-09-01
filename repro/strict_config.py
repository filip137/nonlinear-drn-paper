"""Strict, deterministic configuration loading and composition.

This module is intentionally independent from the experiment runners.  It gives
every scientific entry point the same rules for parsing, validation, reference
resolution, composition, overrides, serialization, and hashing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TypeAlias

import jsonschema
from jsonschema.exceptions import best_match
from referencing import Registry, Resource


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
SchemaSelector: TypeAlias = str | Path | Mapping[str, Any]

_SCHEMA_DIRECTORY = Path("configs/schema")
_SHA256_LENGTH = 64


class ConfigurationError(ValueError):
    """Base class for actionable configuration failures."""


class ConfigurationParseError(ConfigurationError):
    """A JSON file is malformed, ambiguous, or contains a non-finite number."""


class ConfigurationValidationError(ConfigurationError):
    """A parsed document does not satisfy its selected JSON Schema."""


class ConfigurationReferenceError(ConfigurationError):
    """A path-and-digest reference cannot be resolved exactly."""


class ConfigurationCompositionError(ConfigurationError):
    """Configuration sources claim the same section or violate ownership."""


class ConfigurationOverrideError(ConfigurationError):
    """A JSON Pointer override is invalid or cannot be applied."""


@dataclass(frozen=True)
class ConfigReference:
    """An immutable repository-relative reference to exact file bytes."""

    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConfigReference":
        if not isinstance(value, Mapping):
            raise ConfigurationReferenceError(
                f"Expected a config reference object at $, got {type(value).__name__}."
            )
        unknown = sorted(set(value) - {"path", "sha256"})
        missing = sorted({"path", "sha256"} - set(value))
        if missing:
            raise ConfigurationReferenceError(
                f"Expected config reference fields {missing} at $."
            )
        if unknown:
            raise ConfigurationReferenceError(
                f"Unknown config reference fields at $: {unknown}."
            )
        path = value["path"]
        digest = value["sha256"]
        if not isinstance(path, str) or not path.strip():
            raise ConfigurationReferenceError(
                f"Expected config reference $.path to be a non-empty string, got {path!r}."
            )
        if not isinstance(digest, str) or not _is_sha256(digest):
            raise ConfigurationReferenceError(
                "Expected config reference $.sha256 to be 64 lowercase hexadecimal "
                f"characters, got {digest!r}."
            )
        return cls(path=path, sha256=digest)

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ResolvedReference:
    """A verified reference together with its parsed document and absolute path."""

    reference: ConfigReference
    absolute_path: Path
    document: dict[str, Any]

    @property
    def sha256(self) -> str:
        return self.reference.sha256


@dataclass(frozen=True)
class ReferenceRecord:
    """Provenance for one section inserted during composition."""

    owner: str
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"owner": self.owner, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class CompositionResult:
    """A composed document and the exact sources used to create it."""

    document: dict[str, Any]
    references: tuple[ReferenceRecord, ...]


@dataclass(frozen=True)
class JsonPointerOverride:
    """One explicit JSON Pointer replacement."""

    pointer: str
    value: JsonValue


def repository_root() -> Path:
    """Return the repository containing this module."""

    return Path(__file__).resolve().parents[1]


def load_json(path: Path | str) -> Any:
    """Load strict JSON, rejecting duplicate keys and NaN/Infinity constants."""

    candidate = Path(path)
    try:
        text = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationParseError(
            f"Could not read JSON configuration {candidate}: {exc}."
        ) from exc
    return _parse_json_text(text, source=candidate)


def _parse_json_text(text: str, *, source: Path) -> Any:
    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ConfigurationParseError as exc:
        raise ConfigurationParseError(f"Invalid JSON in {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationParseError(
            f"Invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}."
        ) from exc
    _reject_non_finite(document)
    return document


def validate_document(
    document: Any,
    schema: SchemaSelector,
    *,
    repo_root: Path | str | None = None,
) -> None:
    """Validate a document with a local JSON Schema Draft 2020-12 schema.

    ``schema`` may be a schema mapping, a path, a schema filename such as
    ``"training-v2.schema.json"``, or the ``$id`` of a local schema.
    """

    _reject_non_finite(document)
    root = _resolved_root(repo_root)
    schema_document = _load_schema(schema, root)
    validator_class = jsonschema.validators.validator_for(schema_document)
    try:
        validator_class.check_schema(schema_document)
    except jsonschema.SchemaError as exc:
        raise ConfigurationValidationError(
            f"Invalid configuration schema {_schema_label(schema)}: {exc.message}."
        ) from exc
    validator = validator_class(
        schema_document,
        registry=_schema_registry(root),
        format_checker=jsonschema.FormatChecker(),
    )
    errors = list(validator.iter_errors(document))
    if not errors:
        return
    # Resolve each union before comparing top-level errors. ``best_match`` can
    # otherwise descend into an unrelated branch (for example a PWL method
    # error for a malformed Shockley updater).
    candidates = [_most_specific_error(item) for item in errors]
    error = max(candidates, key=_actionable_error_score)
    location = _format_json_path(error.absolute_path)
    raise ConfigurationValidationError(
        f"Configuration does not satisfy {_schema_label(schema)} at {location}: "
        f"{error.message}."
    )


def load_validated_json(
    path: Path | str,
    schema: SchemaSelector,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load a JSON object strictly and validate it before returning it."""

    document = load_json(path)
    if not isinstance(document, dict):
        raise ConfigurationValidationError(
            f"Expected a JSON object at $ in {Path(path)}, got {type(document).__name__}."
        )
    validate_document(document, schema, repo_root=repo_root)
    return document


def canonical_json_bytes(document: Any) -> bytes:
    """Serialize JSON deterministically for hashing and byte-for-byte comparison."""

    _reject_non_finite(document)
    try:
        return json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfigurationParseError(
            f"Expected a JSON-compatible value for canonical serialization: {exc}."
        ) from exc


def canonical_json_text(document: Any) -> str:
    """Return the UTF-8 canonical representation as text."""

    return canonical_json_bytes(document).decode("utf-8")


def pretty_json_text(document: Any) -> str:
    """Return deterministic, human-readable JSON terminated by one newline."""

    _reject_non_finite(document)
    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationParseError(
            f"Expected a JSON-compatible value for serialization: {exc}."
        ) from exc


def document_sha256(document: Any) -> str:
    """Hash a document's canonical JSON representation."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def file_sha256(path: Path | str) -> str:
    """Hash exact file bytes."""

    digest = hashlib.sha256()
    candidate = Path(path)
    try:
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ConfigurationReferenceError(f"Could not hash {candidate}: {exc}.") from exc
    return digest.hexdigest()


def make_reference(path: Path | str, *, repo_root: Path | str | None = None) -> ConfigReference:
    """Create a repository-relative reference from an existing file."""

    root = _resolved_root(repo_root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    _require_inside_repository(candidate, root)
    if not candidate.is_file():
        raise ConfigurationReferenceError(
            f"Expected reference target to be an existing file: {candidate}."
        )
    return ConfigReference(
        path=candidate.relative_to(root).as_posix(),
        sha256=file_sha256(candidate),
    )


def resolve_reference(
    reference: ConfigReference | Mapping[str, Any],
    *,
    repo_root: Path | str | None = None,
    schema: SchemaSelector | None = None,
) -> ResolvedReference:
    """Resolve, hash-check, strictly parse, and optionally validate a config reference."""

    parsed = (
        reference
        if isinstance(reference, ConfigReference)
        else ConfigReference.from_mapping(reference)
    )
    root = _resolved_root(repo_root)
    relative = Path(parsed.path)
    if relative.is_absolute():
        raise ConfigurationReferenceError(
            f"Expected config reference $.path to be repository-relative, got {parsed.path!r}."
        )
    candidate = (root / relative).resolve()
    _require_inside_repository(candidate, root)
    if not candidate.is_file():
        raise ConfigurationReferenceError(
            f"Config reference target does not exist or is not a file: {parsed.path}."
        )
    try:
        raw = candidate.read_bytes()
    except OSError as exc:
        raise ConfigurationReferenceError(
            f"Could not read config reference {parsed.path}: {exc}."
        ) from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != parsed.sha256:
        raise ConfigurationReferenceError(
            f"SHA-256 mismatch for config reference {parsed.path}: expected "
            f"{parsed.sha256}, got {actual}."
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ConfigurationParseError(
            f"Could not decode JSON configuration {candidate}: {exc}."
        ) from exc
    document = _parse_json_text(text, source=candidate)
    if not isinstance(document, dict):
        raise ConfigurationReferenceError(
            f"Expected referenced configuration {parsed.path} to contain an object at $."
        )
    if schema is not None:
        validate_document(document, schema, repo_root=root)
    return ResolvedReference(
        reference=parsed,
        absolute_path=candidate,
        document=document,
    )


def compose_owned_sections(
    base: Mapping[str, Any],
    sources: Mapping[str, ConfigReference | Mapping[str, Any]],
    *,
    repo_root: Path | str | None = None,
    schemas: Mapping[str, SchemaSelector] | None = None,
) -> CompositionResult:
    """Insert one section from each verified, non-overlapping owner source.

    A source assigned to ``"simulation"`` must contain a top-level
    ``"simulation"`` section.  The base document must not already contain that
    section.  References are resolved in sorted owner order so results do not
    depend on mapping insertion order.
    """

    if not isinstance(base, Mapping):
        raise ConfigurationCompositionError(
            f"Expected composition base to be an object at $, got {type(base).__name__}."
        )
    _reject_non_finite(base)
    if not isinstance(sources, Mapping) or not sources:
        raise ConfigurationCompositionError(
            "Expected at least one owned reference in composition sources."
        )
    unknown_schema_owners = set(schemas or {}) - set(sources)
    if unknown_schema_owners:
        raise ConfigurationCompositionError(
            "Schema selectors were supplied for owners without sources: "
            f"{sorted(unknown_schema_owners)}."
        )

    composed = copy.deepcopy(dict(base))
    records: list[ReferenceRecord] = []
    for owner in sorted(sources):
        if not isinstance(owner, str) or not owner:
            raise ConfigurationCompositionError(
                f"Expected every source owner to be a non-empty string, got {owner!r}."
            )
        if owner in composed:
            raise ConfigurationCompositionError(
                f"Configuration ownership collision at $.{owner}: the base and "
                f"reference source both define {owner!r}."
            )
        resolved = resolve_reference(
            sources[owner],
            repo_root=repo_root,
            schema=(schemas or {}).get(owner),
        )
        if owner not in resolved.document:
            raise ConfigurationCompositionError(
                f"Referenced source {resolved.reference.path} owns {owner!r} but "
                f"does not define $.{owner}."
            )
        composed[owner] = copy.deepcopy(resolved.document[owner])
        records.append(
            ReferenceRecord(
                owner=owner,
                path=resolved.reference.path,
                sha256=resolved.reference.sha256,
            )
        )
    return CompositionResult(document=composed, references=tuple(records))


def resolve_config_source(
    source: Path | str | Mapping[str, Any],
    *,
    schema: SchemaSelector,
    reference_fields: Mapping[str, str],
    repo_root: Path | str | None = None,
    reference_schemas: Mapping[str, SchemaSelector] | None = None,
    expanded_schema: SchemaSelector | None = None,
) -> CompositionResult:
    """Validate, compose, provenance-stamp, and revalidate a versioned source.

    ``reference_fields`` maps source fields to the section they own, for
    example ``{"simulation_ref": "simulation", "execution_ref": "execution"}``.
    A fully expanded snapshot with none of those fields is accepted without
    looking up profile files. Partial composition is rejected.
    """

    root = _resolved_root(repo_root)
    if isinstance(source, Mapping):
        document = copy.deepcopy(dict(source))
        validate_document(document, schema, repo_root=root)
    else:
        document = load_validated_json(source, schema, repo_root=root)

    fields = list(reference_fields)
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise ConfigurationCompositionError(
            "Expected reference_fields to map non-empty source fields to owners."
        )
    owners = list(reference_fields.values())
    if any(not isinstance(owner, str) or not owner for owner in owners):
        raise ConfigurationCompositionError(
            "Expected reference_fields to map source fields to non-empty owners."
        )
    duplicate_owners = sorted({owner for owner in owners if owners.count(owner) > 1})
    if duplicate_owners:
        raise ConfigurationCompositionError(
            f"Multiple reference fields claim the same owners: {duplicate_owners}."
        )

    present = [field for field in fields if field in document]
    target_schema = schema if expanded_schema is None else expanded_schema
    if not present:
        validate_document(document, target_schema, repo_root=root)
        return CompositionResult(document=document, references=())
    if set(present) != set(fields):
        missing = sorted(set(fields) - set(present))
        raise ConfigurationCompositionError(
            f"Partially composed configuration is missing reference fields: {missing}."
        )

    sources: dict[str, ConfigReference] = {}
    for field, owner in reference_fields.items():
        value = document.pop(field)
        if not isinstance(value, Mapping):
            raise ConfigurationReferenceError(
                f"Expected $.{field} to contain a path-and-SHA reference object."
            )
        sources[owner] = ConfigReference.from_mapping(value)
    result = compose_owned_sections(
        document,
        sources,
        repo_root=root,
        schemas=reference_schemas,
    )

    provenance = result.document.get("provenance")
    if not isinstance(provenance, dict):
        raise ConfigurationCompositionError(
            "Expected a mutable $.provenance object before source composition."
        )
    if "config_sources" in provenance:
        raise ConfigurationCompositionError(
            "Source configuration already defines $.provenance.config_sources; "
            "the resolver owns this generated field."
        )
    provenance["config_sources"] = [record.as_dict() for record in result.references]
    validate_document(result.document, target_schema, repo_root=root)
    return result


def merge_disjoint_mappings(
    *documents: Mapping[str, Any], labels: Sequence[str] | None = None
) -> dict[str, Any]:
    """Deep-copy and merge mappings only when their top-level keys are disjoint."""

    if labels is not None and len(labels) != len(documents):
        raise ConfigurationCompositionError(
            "Expected one composition label per document."
        )
    names = list(labels) if labels is not None else [f"source[{i}]" for i in range(len(documents))]
    merged: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for name, document in zip(names, documents):
        if not isinstance(document, Mapping):
            raise ConfigurationCompositionError(
                f"Expected {name} to be an object, got {type(document).__name__}."
            )
        overlap = sorted(set(merged).intersection(document))
        if overlap:
            first = {key: owners[key] for key in overlap}
            raise ConfigurationCompositionError(
                f"Configuration ownership collision for fields {overlap}: "
                f"already owned by {first}, repeated by {name}."
            )
        for key, value in document.items():
            merged[key] = copy.deepcopy(value)
            owners[key] = name
    _reject_non_finite(merged)
    return merged


def apply_json_pointer_overrides(
    document: Any,
    overrides: Mapping[str, JsonValue] | Sequence[JsonPointerOverride],
    *,
    require_existing: bool = True,
) -> Any:
    """Apply unambiguous RFC 6901 JSON Pointer replacements to a deep copy.

    Parent/child pointer pairs are rejected because their result would depend on
    application order.  By default, every target must already exist.
    """

    entries = _normalise_overrides(overrides)
    decoded = [(entry, _decode_json_pointer(entry.pointer)) for entry in entries]
    _reject_overlapping_pointers(decoded)
    result = copy.deepcopy(document)
    for entry, tokens in decoded:
        _reject_non_finite(entry.value, path=entry.pointer or "$")
        if not tokens:
            result = copy.deepcopy(entry.value)
            continue
        parent = result
        for depth, token in enumerate(tokens[:-1]):
            parent = _pointer_child(
                parent,
                token,
                pointer=entry.pointer,
                depth=depth,
            )
        _replace_pointer_value(
            parent,
            tokens[-1],
            entry.value,
            pointer=entry.pointer,
            require_existing=require_existing,
        )
    _reject_non_finite(result)
    return result


def _load_schema(schema: SchemaSelector, root: Path) -> dict[str, Any]:
    if isinstance(schema, Mapping):
        return copy.deepcopy(dict(schema))
    path = _resolve_schema_path(schema, root)
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise ConfigurationValidationError(
            f"Expected schema {path} to contain an object at $."
        )
    return loaded


def _resolve_schema_path(schema: str | Path, root: Path) -> Path:
    candidate = Path(schema)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        direct = (root / candidate).resolve()
        if direct.is_file():
            return direct
        schema_dir = root / _SCHEMA_DIRECTORY
        by_name = schema_dir / candidate
        if by_name.is_file():
            return by_name
        if candidate.suffix != ".json":
            with_suffix = schema_dir / f"{candidate}.schema.json"
            if with_suffix.is_file():
                return with_suffix
    selector = str(schema)
    for path in sorted((root / _SCHEMA_DIRECTORY).glob("*.schema.json")):
        document = load_json(path)
        if isinstance(document, dict) and document.get("$id") == selector:
            return path
    raise ConfigurationValidationError(
        f"Could not resolve local configuration schema {selector!r} under "
        f"{root / _SCHEMA_DIRECTORY}."
    )


def _schema_registry(root: Path) -> Registry[Any]:
    registry: Registry[Any] = Registry()
    schema_paths = sorted((root / "configs").rglob("*schema.json"))
    for path in schema_paths:
        document = load_json(path)
        if not isinstance(document, dict):
            continue
        try:
            resource = Resource.from_contents(document)
        except Exception as exc:
            raise ConfigurationValidationError(
                f"Could not register local schema {path}: {exc}."
            ) from exc
        registry = registry.with_resource(path.resolve().as_uri(), resource)
        schema_id = document.get("$id")
        if isinstance(schema_id, str) and schema_id:
            registry = registry.with_resource(schema_id, resource)
    return registry


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationParseError(f"duplicate object key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ConfigurationParseError(f"non-finite numeric constant {value!r} is not permitted.")


def _reject_non_finite(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationParseError(
            f"Non-finite numeric value at {path} is not valid reproducible JSON: {value!r}."
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigurationParseError(
                    f"Expected a string object key at {path}, got {key!r}."
                )
            _reject_non_finite(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, path=f"{path}[{index}]")


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _resolved_root(value: Path | str | None) -> Path:
    root = repository_root() if value is None else Path(value)
    return root.expanduser().resolve()


def _require_inside_repository(candidate: Path, root: Path) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ConfigurationReferenceError(
            f"Config reference escapes repository root {root}: {candidate}."
        ) from exc


def _schema_label(schema: SchemaSelector) -> str:
    if isinstance(schema, Mapping):
        title = schema.get("title") or schema.get("$id")
        return repr(title) if title else "the supplied schema"
    return repr(str(schema))


def _format_json_path(path: Sequence[Any]) -> str:
    result = "$"
    for item in path:
        if isinstance(item, int):
            result += f"[{item}]"
        elif isinstance(item, str) and item.isidentifier():
            result += f".{item}"
        else:
            result += f"[{item!r}]"
    return result


def _most_specific_error(error: jsonschema.ValidationError) -> jsonschema.ValidationError:
    current = error
    seen: set[int] = set()
    while current.context and id(current) not in seen:
        seen.add(id(current))
        if current.validator in {"oneOf", "anyOf"}:
            nested = _best_discriminated_branch_error(current.context)
        else:
            nested = best_match(current.context)
        if nested is None or nested is current:
            break
        current = nested
    return current


def _best_discriminated_branch_error(
    errors: Sequence[jsonschema.ValidationError],
) -> jsonschema.ValidationError | None:
    """Prefer errors from the union branch that best matches its discriminators."""

    if not errors:
        return None
    branches: dict[Any, list[jsonschema.ValidationError]] = {}
    for error in errors:
        relative_schema_path = list(error.relative_schema_path)
        branch = relative_schema_path[0] if relative_schema_path else None
        branches.setdefault(branch, []).append(error)
    selected = min(
        branches.values(),
        key=lambda branch_errors: (
            sum(_branch_error_cost(item) for item in branch_errors),
            len(branch_errors),
        ),
    )
    return max(selected, key=_actionable_error_score)


def _branch_error_cost(error: jsonschema.ValidationError) -> int:
    weights = {
        "const": 1000,
        "enum": 1000,
        "required": 100,
        "additionalProperties": 10,
        "type": 10,
    }
    return weights.get(str(error.validator), 25)


def _actionable_error_score(error: jsonschema.ValidationError) -> tuple[int, int]:
    weights = {
        "type": 50,
        "additionalProperties": 40,
        "required": 30,
        "enum": 20,
        "const": 10,
    }
    return (
        weights.get(str(error.validator), 0),
        len(error.absolute_path),
    )


def _normalise_overrides(
    overrides: Mapping[str, JsonValue] | Sequence[JsonPointerOverride],
) -> list[JsonPointerOverride]:
    if isinstance(overrides, Mapping):
        invalid = [pointer for pointer in overrides if not isinstance(pointer, str)]
        if invalid:
            raise ConfigurationOverrideError(
                f"Expected JSON Pointer override keys to be strings, got {invalid!r}."
            )
        entries = [JsonPointerOverride(pointer, value) for pointer, value in overrides.items()]
    else:
        entries = list(overrides)
        if any(not isinstance(entry, JsonPointerOverride) for entry in entries):
            raise ConfigurationOverrideError(
                "Expected override sequence entries to be JsonPointerOverride objects."
            )
    pointers = [entry.pointer for entry in entries]
    duplicates = sorted({pointer for pointer in pointers if pointers.count(pointer) > 1})
    if duplicates:
        raise ConfigurationOverrideError(f"Duplicate JSON Pointer overrides: {duplicates}.")
    return entries


def _decode_json_pointer(pointer: str) -> tuple[str, ...]:
    if not isinstance(pointer, str):
        raise ConfigurationOverrideError(
            f"Expected a JSON Pointer string, got {pointer!r}."
        )
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise ConfigurationOverrideError(
            f"Expected JSON Pointer to be empty or start with '/', got {pointer!r}."
        )
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        decoded = ""
        while index < len(raw):
            character = raw[index]
            if character != "~":
                decoded += character
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise ConfigurationOverrideError(
                    f"Invalid '~' escape in JSON Pointer {pointer!r}."
                )
            decoded += "~" if raw[index + 1] == "0" else "/"
            index += 2
        tokens.append(decoded)
    return tuple(tokens)


def _reject_overlapping_pointers(
    decoded: Sequence[tuple[JsonPointerOverride, tuple[str, ...]]],
) -> None:
    for index, (left, left_tokens) in enumerate(decoded):
        for right, right_tokens in decoded[index + 1 :]:
            shortest = min(len(left_tokens), len(right_tokens))
            if left_tokens[:shortest] == right_tokens[:shortest]:
                raise ConfigurationOverrideError(
                    "Overlapping JSON Pointer overrides are order-dependent: "
                    f"{left.pointer!r} and {right.pointer!r}."
                )


def _pointer_child(parent: Any, token: str, *, pointer: str, depth: int) -> Any:
    if isinstance(parent, dict):
        if token not in parent:
            raise ConfigurationOverrideError(
                f"JSON Pointer {pointer!r} does not exist at segment {depth}: {token!r}."
            )
        return parent[token]
    if isinstance(parent, list):
        index = _array_index(token, pointer=pointer)
        if index >= len(parent):
            raise ConfigurationOverrideError(
                f"JSON Pointer {pointer!r} array index {index} is out of range."
            )
        return parent[index]
    raise ConfigurationOverrideError(
        f"JSON Pointer {pointer!r} traverses through scalar value at segment {depth}."
    )


def _replace_pointer_value(
    parent: Any,
    token: str,
    value: JsonValue,
    *,
    pointer: str,
    require_existing: bool,
) -> None:
    copied = copy.deepcopy(value)
    if isinstance(parent, dict):
        if require_existing and token not in parent:
            raise ConfigurationOverrideError(
                f"JSON Pointer override target {pointer!r} does not exist."
            )
        parent[token] = copied
        return
    if isinstance(parent, list):
        index = _array_index(token, pointer=pointer)
        if index < len(parent):
            parent[index] = copied
            return
        if not require_existing and index == len(parent):
            parent.append(copied)
            return
        raise ConfigurationOverrideError(
            f"JSON Pointer {pointer!r} array index {index} is out of range."
        )
    raise ConfigurationOverrideError(
        f"JSON Pointer {pointer!r} cannot replace a child of a scalar value."
    )


def _array_index(token: str, *, pointer: str) -> int:
    if token == "-":
        raise ConfigurationOverrideError(
            f"JSON Pointer {pointer!r} uses '-', which is not allowed for deterministic replacement."
        )
    if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
        raise ConfigurationOverrideError(
            f"Expected a canonical non-negative array index in JSON Pointer {pointer!r}, "
            f"got {token!r}."
        )
    return int(token)


__all__ = [
    "CompositionResult",
    "ConfigReference",
    "ConfigurationCompositionError",
    "ConfigurationError",
    "ConfigurationOverrideError",
    "ConfigurationParseError",
    "ConfigurationReferenceError",
    "ConfigurationValidationError",
    "JsonPointerOverride",
    "ReferenceRecord",
    "ResolvedReference",
    "apply_json_pointer_overrides",
    "canonical_json_bytes",
    "canonical_json_text",
    "compose_owned_sections",
    "document_sha256",
    "file_sha256",
    "load_json",
    "load_validated_json",
    "make_reference",
    "merge_disjoint_mappings",
    "pretty_json_text",
    "repository_root",
    "resolve_reference",
    "resolve_config_source",
    "validate_document",
]
