"""Doc catalog: a short summary per document, so Gate 1 knows what the library covers
(no hard-coded topic list; new documents are covered automatically)."""
from langchain_core.messages import HumanMessage, SystemMessage

from rag.providers.llm import chat_model

SUMMARY_PROMPT = """Summarise what this document covers in 2-3 sentences for a library catalogue.
Name the subject area, the kinds of information inside (rules, tables, part lists, processes,
requirements) and the key standards, products or codes it mentions. No opinions."""


def summarize(title: str, sections: list[str], opening: str) -> str:
    msg = (f"Title: {title}\n\nSections:\n" + "\n".join(sorted(set(sections))[:60]) +
           f"\n\nOpening text:\n{opening[:8000]}")
    return chat_model(small=True).invoke([SystemMessage(SUMMARY_PROMPT), HumanMessage(msg)]).content.strip()
