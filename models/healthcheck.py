from __future__ import annotations
import requests
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

def ensure_models(base_url: str, chat_model: str, embed_model: str, verbose: bool = True) -> None:
    # tags
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()
    tags = [m["name"] for m in r.json().get("models", [])]
    missing = [m for m in (chat_model, embed_model) if m not in tags]
    if missing:
        raise RuntimeError(f"Missing models: {missing}. Pull them via: ollama pull <model>")

    # ping chat
    llm = Ollama(model=chat_model, base_url=base_url)
    resp = llm.invoke("Hello, are you online?")
    if not resp:
        raise RuntimeError(f"{chat_model} did not respond")
    if verbose:
        print(f"✅ {chat_model} replied.")

    # ping embeddings
    emb = OllamaEmbeddings(model=embed_model, base_url=base_url)
    vec = emb.embed_query("test")
    if not vec:
        raise RuntimeError(f"{embed_model} returned empty vector")
    if verbose:
        print(f"✅ {embed_model} embedding OK (len={len(vec)}).")
