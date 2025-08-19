from __future__ import annotations
from typing import List, Set

from .semantic import SemanticMemory
from .chrono import ChronoMemory

class MemoryManager:
    def __init__(self, semantic: SemanticMemory, chrono: ChronoMemory, last_n: int = 6, top_k: int = 5) -> None:
        self.semantic = semantic
        self.chrono = chrono
        self.last_n_n = last_n
        self.top_k = top_k

    def build_context(self, user_query: str) -> str:
        recent = self.chrono.last_n(self.last_n_n)
        sem = self.semantic.retrieve(user_query, k=self.top_k)

        lines: List[str] = []
        # chronological
        for role, text in recent:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")

        # semantic
        if sem:
            lines.append("--- relevant memory ---")
            for s in sem:
                lines.append(f"- {s}")

        return "\n".join(lines)
