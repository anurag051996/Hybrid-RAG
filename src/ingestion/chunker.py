"""
src/ingestion/chunker.py
─────────────────────────
Splits documents into overlapping chunks using a recursive character splitter.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from src.ingestion.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk derived from a Document."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """
    Splits documents into fixed-size overlapping chunks.

    Tries to split on paragraph → sentence → word boundaries
    before hard-splitting on characters.

    Usage:
        chunker = RecursiveChunker(chunk_size=512, overlap=64)
        chunks = chunker.split(documents)
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: List[Document]) -> List[Chunk]:
        """Split a list of Documents into Chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            raw_chunks = self._split_text(doc.page_content)
            for i, text in enumerate(raw_chunks):
                all_chunks.append(Chunk(
                    chunk_id=f"{doc.metadata.get('source', 'doc')}::chunk_{i}",
                    text=text,
                    metadata={**doc.metadata, "chunk_index": i}
                ))
        logger.info("Created %d chunks from %d documents", len(all_chunks), len(documents))
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursive split: tries each separator in order."""
        return self._recursive_split(text, self.SEPARATORS)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:]

        if sep == "" or sep not in text:
            # Hard split by character
            return self._hard_split(text)

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If single part is too big, recurse with next separator
                if len(part) > self.chunk_size and remaining_seps:
                    chunks.extend(self._recursive_split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return self._add_overlap(chunks)

    def _hard_split(self, text: str) -> List[str]:
        """Hard split by character with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.overlap
        return [c for c in chunks if c]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add trailing overlap from previous chunk to current chunk."""
        if len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.overlap:] if self.overlap else ""
            overlapped.append((prev_tail + " " + chunks[i]).strip())
        return overlapped
