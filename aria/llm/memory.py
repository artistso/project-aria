"""A text memory ARIA keeps for itself, and the rule that keeps it honest.

A language model's knowledge is in its weights: it is there, but it cannot be
looked at, corrected, or cited. This module gives the system a second place to
keep what it has learned -- a file of short text entries it can search on
purpose, the way a person keeps notes and rereads them.

Three kinds of entry:

``fact``
    A statement that is true or false: *the Schwarzschild horizon is a null
    surface at r = 2M*, *a cube has 6 faces*. Facts need provenance.
``procedure``
    Something that worked: *to solve a linear equation, move everything to one
    side and isolate the variable*. Procedures come from the ledger.
``correction``
    Something believed before that turned out wrong. Corrections are the most
    valuable entries, because they are the only ones that cannot be re-derived
    from a textbook.

The rule that keeps this from becoming a diary: **every entry carries a source
and a confidence.** An entry with no source is remembered as ``unverified`` and
:meth:`Memory.audit` reports it. Nothing is allowed into memory pretending to be
knowledge when it is only a guess -- that is how a text memory becomes
confabulation.

Memory is used three ways:

1. **Retrieval** -- :meth:`Memory.context` builds a compact block of relevant
   entries to put in front of the model. This is the "look into it" part.
2. **Curriculum** -- procedures point at the ledger entries that proved them, so
   studying follows what has actually been learned.
3. **Rewrite** -- a memory that is retrieved often and never contradicted is the
   best candidate for distillation into weights (see ``docs/MEMORY.md``).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_PATH",
    "KINDS",
    "CONFIDENCES",
    "Memory",
    "MemoryStore",
    "tokenize",
]

#: Where memories live. ``ARIA_MEMORY`` overrides it.
DEFAULT_PATH = Path(
    os.environ.get("ARIA_MEMORY", str(Path.home() / ".aria" / "memory.jsonl"))
)

KINDS = ("fact", "procedure", "correction", "source")

#: Ordered weakest to strongest. Anything below ``tested`` is flagged by audit.
CONFIDENCES = ("unverified", "stated", "tested", "verified")

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "for",
    "on", "at", "by", "with", "that", "this", "it", "as", "be", "from", "if",
}


def tokenize(text: str) -> Tuple[str, ...]:
    """Lowercase word tokens. Deliberately lexical: no model, so no surprises."""
    return tuple(
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in _STOP and len(token) > 1
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Memory:
    """One thing remembered."""

    id: int
    text: str
    kind: str = "fact"
    tags: Tuple[str, ...] = ()
    source: str = ""
    confidence: str = "unverified"
    links: Tuple[str, ...] = ()
    when: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown memory kind {self.kind!r}; expected one of {KINDS}")
        if self.confidence not in CONFIDENCES:
            raise ValueError(
                f"unknown confidence {self.confidence!r}; expected one of {CONFIDENCES}"
            )
        if not self.when:
            self.when = _now()

    @property
    def key(self) -> str:
        """Normalised identity, for de-duplication."""
        return " ".join(tokenize(self.text))

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "when": self.when,
            "text": self.text,
            "kind": self.kind,
            "tags": list(self.tags),
            "source": self.source,
            "confidence": self.confidence,
            "links": list(self.links),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(
            id=int(data.get("id", 0)),
            text=str(data.get("text", "")),
            kind=str(data.get("kind", "fact")),
            tags=tuple(data.get("tags", ())),
            source=str(data.get("source", "")),
            confidence=str(data.get("confidence", "unverified")),
            links=tuple(data.get("links", ())),
            when=str(data.get("when", "")),
        )

    def line(self) -> str:
        """One line for a prompt: what is remembered, and how well founded it is."""
        tags = f" [{', '.join(self.tags)}]" if self.tags else ""
        origin = f" ({self.source})" if self.source else ""
        return f"- ({self.kind}, {self.confidence}) {self.text}{tags}{origin}"

    def markdown(self) -> str:
        tags = f" `{', '.join(self.tags)}`" if self.tags else ""
        origin = self.source or "no source"
        return f"- **{self.kind}** · *{self.confidence}* · {self.text}{tags} — _{origin}_"


class MemoryStore:
    """An append-only file of memories plus lexical retrieval over it."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH

    # -- reading -----------------------------------------------------------

    def entries(self) -> List[Memory]:
        if not self.path.exists():
            return []
        out: List[Memory] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Memory.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
        return out

    def next_id(self) -> int:
        entries = self.entries()
        return (max(entry.id for entry in entries) + 1) if entries else 1

    # -- writing -----------------------------------------------------------

    def add(
        self,
        text: str,
        kind: str = "fact",
        tags: Sequence[str] = (),
        source: str = "",
        confidence: str = "unverified",
        links: Sequence[str] = (),
    ) -> Memory:
        """Remember something. Identical text is not stored twice."""
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("refusing to remember an empty memory")

        memory = Memory(
            id=0,
            text=cleaned,
            kind=kind,
            tags=tuple(tag.strip().lower() for tag in tags if tag.strip()),
            source=source.strip(),
            confidence=confidence,
            links=tuple(links),
        )
        for existing in self.entries():
            if existing.key == memory.key:
                return existing

        memory.id = self.next_id()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(memory.as_dict(), ensure_ascii=False) + "\n")
        return memory

    def forget(self, memory_id: int) -> bool:
        """Remove one entry by id. Rewrites the file atomically."""
        entries = self.entries()
        kept = [entry for entry in entries if entry.id != memory_id]
        if len(kept) == len(entries):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            "".join(json.dumps(e.as_dict(), ensure_ascii=False) + "\n" for e in kept),
            encoding="utf-8",
        )
        tmp.replace(self.path)
        return True

    # -- retrieval ---------------------------------------------------------

    def find(self, query: str, k: int = 5) -> List[Memory]:
        """Rank memories against a query with an IDF-weighted lexical score.

        No embedding model: the ranking has to be explainable and reproducible
        on a tablet with no GPU. Ties break toward stronger confidence, then
        toward the more recent entry.
        """
        entries = self.entries()
        if not entries:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return entries[-k:][::-1]

        document_frequency: Dict[str, int] = {}
        tokenised = []
        for entry in entries:
            tokens = set(tokenize(" ".join((entry.text, " ".join(entry.tags)))))
            tokenised.append(tokens)
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        total = len(entries)
        scored: List[Tuple[float, Memory]] = []
        for entry, tokens in zip(entries, tokenised):
            overlap = query_tokens & tokens
            if not overlap:
                continue
            score = sum(
                math.log(1.0 + total / document_frequency[token]) for token in overlap
            )
            score *= 1.0 + 0.5 * len(overlap) / len(query_tokens)  # coverage bonus
            # A word the author chose as a tag is a stronger signal than the same
            # word occurring in prose, so tag hits are weighted separately.
            score *= 1.0 + 0.5 * len(query_tokens & set(entry.tags))
            score *= 1.0 + 0.25 * CONFIDENCES.index(entry.confidence)
            scored.append((score, entry))

        scored.sort(key=lambda pair: (-pair[0], -pair[1].id))
        return [entry for _, entry in scored[:k]]

    def context(self, query: str, k: int = 6, max_chars: int = 1600) -> str:
        """A block of relevant memory, ready to put in front of a model."""
        hits = self.find(query, k=k)
        if not hits:
            return ""
        lines = [
            f"What ARIA remembers that bears on {query!r} "
            f"({len(hits)} of {len(self.entries())} entries):",
        ]
        used = len(lines[0])
        for memory in hits:
            line = memory.line()
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        lines.append(
            "Use an entry only if it applies; each carries its source, and "
            "`unverified` means nobody has checked it."
        )
        return "\n".join(lines)

    # -- reporting ---------------------------------------------------------

    def digest(self) -> dict:
        entries = self.entries()
        by_kind: Dict[str, int] = {kind: 0 for kind in KINDS}
        by_confidence: Dict[str, int] = {level: 0 for level in CONFIDENCES}
        tags: Dict[str, int] = {}
        for entry in entries:
            by_kind[entry.kind] += 1
            by_confidence[entry.confidence] += 1
            for tag in entry.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return {
            "entries": len(entries),
            "by_kind": by_kind,
            "by_confidence": by_confidence,
            "tags": dict(sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))),
            "unverified": [entry.id for entry in entries if entry.confidence == "unverified"],
            "unsourced": [entry.id for entry in entries if not entry.source],
        }

    def audit(self) -> List[Memory]:
        """Entries that claim to be knowledge without anything holding them up."""
        return [
            entry
            for entry in self.entries()
            if not entry.source or entry.confidence == "unverified"
        ]

    def render_markdown(self, title: str = "What ARIA remembers") -> str:
        """The human-readable face of the memory file."""
        entries = self.entries()
        digest = self.digest()
        lines = [
            f"# {title}",
            "",
            f"{digest['entries']} entries · "
            + " · ".join(f"{kind}: {count}" for kind, count in digest["by_kind"].items()),
            "",
        ]
        for kind in KINDS:
            group = [entry for entry in entries if entry.kind == kind]
            if not group:
                continue
            lines.append(f"## {kind}")
            lines.append("")
            lines.extend(entry.markdown() for entry in group)
            lines.append("")
        flagged = self.audit()
        if flagged:
            lines.append("## needs a source")
            lines.append("")
            lines.extend(f"- #{entry.id} {entry.text}" for entry in flagged)
            lines.append("")
        return "\n".join(lines)

    # -- feeding memory from work -----------------------------------------

    def import_ledger(self, ledger, only_demonstrated: bool = True) -> List[Memory]:
        """Turn proven ledger entries into procedures, and failures into corrections."""
        added: List[Memory] = []
        for entry in ledger.entries():
            if entry.demonstrated and only_demonstrated:
                text = (
                    f"Can prove `{entry.contract.splitlines()[-1]}` "
                    f"({entry.skill}); asked as {entry.question!r}"
                )
                added.append(
                    self.add(
                        text,
                        kind="procedure",
                        tags=(entry.skill, "ledger"),
                        source=f"ledger#{entry.seq}",
                        confidence="verified",
                        links=(f"ledger#{entry.seq}",),
                    )
                )
            elif not entry.demonstrated and entry.first_error:
                added.append(
                    self.add(
                        f"Attempt on {entry.question!r} broke at {entry.first_error}",
                        kind="correction",
                        tags=(entry.skill, "ledger"),
                        source=f"ledger#{entry.seq}",
                        confidence="tested",
                        links=(f"ledger#{entry.seq}",),
                    )
                )
        return added

    def import_geometry(self) -> List[Memory]:
        """Seed memory with shape properties and causal structure, all verified.

        Geometry is a good first tenant for a text memory because every claim is
        checkable: a count, an angle, a coordinate identity. If a remembered
        shape fact is ever wrong, ``aria geometry check`` says so.
        """
        from ..geometry import PLATONIC, causal_checks, shape_checks, structure

        added: List[Memory] = []
        for solid in PLATONIC:
            added.append(
                self.add(
                    f"{solid.name} {{{solid.face_sides},{solid.faces_at_vertex}}}: "
                    f"V={solid.vertices}, E={solid.edges}, F={solid.faces}, "
                    f"V-E+F={solid.euler_characteristic}, "
                    f"dihedral {solid.dihedral_deg:.4f} deg, dual {solid.dual}, "
                    f"symmetry group order {solid.symmetry_order}",
                    kind="fact",
                    tags=("geometry", "polyhedron", solid.name),
                    source="aria/geometry/shapes.py",
                    confidence="verified",
                )
            )
        for model in ("minkowski", "schwarzschild", "frw"):
            data = structure(model)
            for element in data["elements"]:  # type: ignore[index]
                added.append(
                    self.add(
                        f"{model} Penrose diagram: {element['name']} is "
                        f"{element['kind']} at {element['where']}; {element['means']}",
                        kind="fact",
                        tags=("geometry", "penrose", model),
                        source="aria/geometry/causal.py",
                        confidence="verified",
                        links=(f"geometry:penrose:{model}",),
                    )
                )
        for claim, passed in shape_checks() + causal_checks():
            if passed:
                added.append(
                    self.add(
                        f"checked: {claim}",
                        kind="fact",
                        tags=("geometry", "check"),
                        source="aria geometry check",
                        confidence="verified",
                    )
                )
        return added

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(entry.as_dict(), ensure_ascii=False) + "\n" for entry in self.entries()
        )
