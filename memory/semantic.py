from __future__ import annotations
import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings

class SemanticMemory:
    def __init__(self, embedder: OllamaEmbeddings, store_dir: str = "vector_store") -> None:
        self.embedder = embedder
        self.store_dir = store_dir
        self._load()

    def _load(self) -> None:
        if os.path.isdir(self.store_dir) and any(os.scandir(self.store_dir)):
            self.store = FAISS.load_local(self.store_dir, self.embedder, allow_dangerous_deserialization=True)
        else:
            self.store = FAISS.from_texts(texts=[], embedding=self.embedder)

    def _persist(self) -> None:
        os.makedirs(self.store_dir, exist_ok=True)
        self.store.save_local(self.store_dir)

    def add(self, text: str, metadata: Optional[dict] = None) -> None:
        doc = Document(page_content=text, metadata=metadata or {})
        self.store.add_documents([doc])
        self._persist()

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        docs = self.store.similarity_search(query, k=k)
        return [d.page_content for d in docs]
