"""Original-text windows with offsets and deterministic query-aware MMR ranks."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceWindow:
    start: int
    end: int
    content: str
    score: float
    heading: str | None = None
    char_start: int = -1
    char_end: int = -1


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9][a-z0-9_.-]+", text.casefold()))
    for phrase in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(phrase[i : i + 2] for i in range(max(1, len(phrase) - 1)))
    return words


def rank_source_windows(
    text: str,
    objectives: tuple[str, ...],
    *,
    window_chars: int = 3000,
    limit: int = 64,
) -> list[SourceWindow]:
    if window_chars <= 0 or limit <= 0:
        raise ValueError("positive window size and limit required")
    windows: list[tuple[int, int, str, set[str]]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + window_chars)
        boundary = text.rfind("\n\n", start + window_chars // 2, end)
        if boundary > start and end < len(text):
            end = boundary + 2
        chunk = text[start:end]
        windows.append((start, end, chunk, terms(chunk)))
        start = end
    queries = [terms(q) for q in objectives if q.strip()]
    relevance = [max((len(w[3] & q) / max(1, len(q)) for q in queries), default=0) for w in windows]
    selected: list[int] = []
    remaining = set(range(len(windows)))
    while remaining and len(selected) < limit:

        def score(i: int) -> tuple[float, int]:
            overlap = max(
                (
                    len(windows[i][3] & windows[j][3]) / max(1, len(windows[i][3] | windows[j][3]))
                    for j in selected
                ),
                default=0,
            )
            return (0.7 * relevance[i] - 0.3 * overlap, -i)

        winner = max(remaining, key=score)
        selected.append(winner)
        remaining.remove(winner)
    return [
        SourceWindow(
            windows[i][0], windows[i][1], windows[i][2], 1.0 - rank / max(1, len(selected))
        )
        for rank, i in enumerate(selected)
    ]


def _block_heading(blocks: list[tuple[int, int, str]], index: int) -> str | None:
    """Return the nearest preceding heading line (or the block's own first line)."""
    first_line = blocks[index][2].splitlines()[0] if blocks[index][2].splitlines() else ""
    if _is_heading(first_line):
        return first_line.strip()[:80]
    for i in range(index - 1, -1, -1):
        if _is_heading(blocks[i][2]):
            return blocks[i][2].strip()[:80]
        if i < index - 2:
            break
    return None


def _block_relevance(content: str, queries: list[set[str]]) -> float:
    tokens = terms(content)
    return max((len(tokens & q) / max(1, len(q)) for q in queries), default=0.0)


_BLOCK_BOUNDARY_RE = re.compile(r"(?:\n\s*\n)|\r\n\r\n|(?=\r?\n#{1,6}\s)")


def _split_blocks(text: str) -> list[tuple[int, int, str]]:
    """Split the source on paragraph/title boundaries preserving character offsets."""
    if not text:
        return []
    blocks: list[tuple[int, int, str]] = []
    start = 0
    for match in _BLOCK_BOUNDARY_RE.finditer(text):
        end = match.start()
        if end > start:
            blocks.append((start, end, text[start:end]))
        start = match.end()
    if start < len(text):
        blocks.append((start, len(text), text[start:]))
    return blocks


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"^(?:#\s+)|(?:^\s*(?:=+|:+|-+)\s*$)", stripped, re.MULTILINE):
        return True
    # A short standalone line spanning no sentence-ending punctuation is likely a title.
    sentence_end = ("\u3002", ". ", "\uff1b", ";", "\uff1a", ":")
    return len(stripped) <= 60 and not any(p in stripped for p in sentence_end)


def select_relevant_blocks(
    text: str,
    objectives: tuple[str, ...],
    *,
    limit: int = 64,
    expand_neighbors: int = 0,
) -> list[SourceWindow]:
    """Block-aware relevance selection with table-heading and neighbor support.

    Splits on paragraph/title boundaries (offsets preserved), applies MMR
    de-duplication over the objective terms, optionally expands around each
    selected block so cross-block answers are not split, and returns blocks in
    original-text order with their nearest preceding heading.
    """

    if limit <= 0:
        raise ValueError("positive limit required")
    blocks = _split_blocks(text)
    if not blocks:
        return []
    queries = [terms(q) for q in objectives if q.strip()]
    if not queries:
        return [
            SourceWindow(
                start,
                end,
                content,
                1.0,
                heading=_block_heading(blocks, i),
                char_start=start,
                char_end=end,
            )
            for i, (start, end, content) in enumerate(blocks[:limit])
        ]
    block_terms = [terms(blocks[i][2]) for i in range(len(blocks))]
    relevance = [
        max((len(block_terms[i] & q) / max(1, len(q)) for q in queries), default=0.0)
        for i in range(len(blocks))
    ]

    def overlap(i: int, selected: list[int]) -> float:
        bi = block_terms[i]
        return max(
            (
                len(bi & block_terms[j]) / max(1, len(bi | block_terms[j]))
                for j in selected
            ),
            default=0.0,
        )

    selected: list[int] = []
    remaining = set(range(len(blocks)))
    while remaining and len(selected) < limit:

        def score(i: int) -> tuple[float, int]:
            return (0.7 * relevance[i] - 0.3 * overlap(i, selected), -i)

        winner = max(remaining, key=score)
        selected.append(winner)
        remaining.remove(winner)

    if expand_neighbors > 0:
        expanded = set(selected)
        for i in selected:
            for distance in range(1, int(expand_neighbors) + 1):
                if i - distance >= 0:
                    expanded.add(i - distance)
                if i + distance < len(blocks):
                    expanded.add(i + distance)
        selected = sorted(expanded)

    return [
        SourceWindow(
            blocks[i][0],
            blocks[i][1],
            blocks[i][2],
            relevance[i],
            heading=_block_heading(blocks, i),
            char_start=blocks[i][0],
            char_end=blocks[i][1],
        )
        for i in sorted(selected)
    ]
