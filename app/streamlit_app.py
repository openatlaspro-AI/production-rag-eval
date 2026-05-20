"""Streamlit live demo for production-rag-eval.

Same RAG pipeline as the production FastAPI service (src/api.py), with one
substitution: retrieval runs over pre-computed embeddings in numpy
(src/retrieve_inmemory.py) instead of pgvector, because Streamlit Cloud has no
Postgres. Generation still goes through the live Mistral API.

Reads MISTRAL_API_KEY from st.secrets (Streamlit Cloud) or env (local).
Does NOT import src/config.py — that would fail at boot when the key is missing.
"""

from __future__ import annotations

import os
import time

import streamlit as st

GITHUB_URL = "https://github.com/openatlaspro-AI/production-rag-eval"

# ---- Page config ---------------------------------------------------------
st.set_page_config(
    page_title="Production RAG Eval — Live Demo",
    page_icon="🔎",
    layout="wide",
)

st.title("Production RAG Eval — Live Demo")
st.caption(
    "Production-style RAG over a 400-doc Chinese-news trend corpus. "
    "Same Mistral pipeline as the [FastAPI service]({url}); retrieval runs in-memory "
    "(numpy cosine over pre-computed embeddings) since Streamlit Cloud has no Postgres."
    .format(url=GITHUB_URL)
)


# ---- API key handling ----------------------------------------------------
def get_api_key() -> str | None:
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get("MISTRAL_API_KEY")


api_key = get_api_key()
if not api_key:
    st.warning(
        "⚠️ `MISTRAL_API_KEY` not set.\n\n"
        "**Streamlit Cloud:** Settings → Secrets → add `MISTRAL_API_KEY = \"...\"`.\n\n"
        "**Local:** copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` "
        "and paste your key, or `export MISTRAL_API_KEY=...`."
    )
    st.stop()


# ---- Lazy / cached resource loaders --------------------------------------
@st.cache_resource(show_spinner="Loading Mistral client…")
def get_client(key: str):
    from mistralai import Mistral
    return Mistral(api_key=key)


@st.cache_resource(show_spinner="Loading 400-doc corpus…")
def get_corpus():
    from src.retrieve_inmemory import load_corpus
    return load_corpus()


# Import lazily so the warning above renders before any heavy imports if key missing.
from src.pricing import calc_cost, format_cost  # noqa: E402
from src.retrieve import embed_query  # noqa: E402
from src.retrieve_inmemory import inmemory_search  # noqa: E402

EMBED_MODEL = "mistral-embed"
MODEL_OPTIONS = {
    "mistral-small-latest (default, 10× cheaper)": "mistral-small-latest",
    "mistral-large-latest": "mistral-large-latest",
}

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about trending topics in "
    "Chinese-language media.\n\n"
    "Answer the user's question using ONLY the numbered context items below. The items "
    "are news headlines that have been translated to English. Cite sources inline using "
    "bracket notation: [1], [2], etc., referring to the numbered items.\n\n"
    "Rules:\n"
    "- Use ONLY the provided context. If the answer is not in the context, say so plainly.\n"
    "- Cite every factual claim with the corresponding [N] reference.\n"
    "- Keep the answer concise (3-6 sentences).\n"
    "- Do not invent details not present in the context."
)


def build_user_message(query: str, hits: list) -> str:
    lines = ["Context:"]
    for i, h in enumerate(hits, 1):
        platform = f" (source: {h.platform})" if h.platform else ""
        title = h.title_en or h.title or "(no title)"
        lines.append(f"[{i}] {title}{platform}")
    lines.extend(["", f"Question: {query}", "", "Answer:"])
    return "\n".join(lines)


# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox(
        "Generation model",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="`small` matches `large` on LLM-judge quality at 10× lower cost — see README.",
    )
    model = MODEL_OPTIONS[model_label]
    k = st.slider("Top-k retrieved", min_value=1, max_value=10, value=5)
    st.divider()
    st.markdown(f"[GitHub repo]({GITHUB_URL})")
    st.caption("Corpus: 400 Chinese-news titles (translated → embedded with `mistral-embed`).")


# ---- Main UI -------------------------------------------------------------
default_query = "what trends involve Chinese geopolitics"
query = st.text_input(
    "Your query",
    value=default_query,
    placeholder="e.g. what trends relate to family or household life",
)
search_clicked = st.button("Search", type="primary", disabled=not query.strip())


if search_clicked:
    client = get_client(api_key)
    corpus_emb, docs = get_corpus()

    try:
        with st.spinner("Embedding query…"):
            t0 = time.perf_counter()
            q_emb, embed_tokens_in = embed_query(client, query)
            embed_ms = (time.perf_counter() - t0) * 1000

        with st.spinner(f"Retrieving top {k}…"):
            t1 = time.perf_counter()
            hits = inmemory_search(q_emb, corpus_emb, docs, k=k)
            retrieve_ms = (time.perf_counter() - t1) * 1000

        with st.spinner(f"Generating answer with {model}…"):
            user_msg = build_user_message(query, hits)
            t2 = time.perf_counter()
            chat = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            generate_ms = (time.perf_counter() - t2) * 1000

    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
        st.stop()

    answer = chat.choices[0].message.content
    usage = chat.usage
    gen_tokens_in = usage.prompt_tokens if usage else 0
    gen_tokens_out = usage.completion_tokens if usage else 0
    embed_usd = calc_cost(EMBED_MODEL, embed_tokens_in, 0)
    generate_usd = calc_cost(model, gen_tokens_in, gen_tokens_out)
    total_usd = embed_usd + generate_usd

    # --- Answer
    st.subheader("Answer")
    st.markdown(answer)

    # --- Sources
    st.subheader("Retrieved documents")
    table_rows = [
        {
            "rank": i,
            "similarity": round(h.similarity, 3),
            "platform": h.platform or "—",
            "English title": h.title_en or "",
            "Original (zh)": h.title or "",
        }
        for i, h in enumerate(hits, 1)
    ]
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    # --- Latency
    st.subheader("Latency")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("embed", f"{embed_ms:.0f} ms")
    c2.metric("retrieve", f"{retrieve_ms:.1f} ms")
    c3.metric("generate", f"{generate_ms:.0f} ms")
    c4.metric("total", f"{embed_ms + retrieve_ms + generate_ms:.0f} ms")

    # --- Cost
    st.subheader("Cost")
    c1, c2, c3 = st.columns(3)
    c1.metric("embed", format_cost(embed_usd))
    c2.metric("generate", format_cost(generate_usd))
    c3.metric("total", format_cost(total_usd))
    st.caption(
        f"Tokens — embed-in: {embed_tokens_in} · gen-in: {gen_tokens_in} · gen-out: {gen_tokens_out}"
    )


# ---- Footer --------------------------------------------------------------
st.divider()
st.caption(
    f"[Source on GitHub]({GITHUB_URL}) · "
    "Production path uses pgvector (HNSW) on Postgres 16 + FastAPI. "
    "Eval suite (30-query labeled set, LLM-judge, refusal tests) lives at "
    f"[`eval_results/`]({GITHUB_URL}/tree/main/eval_results)."
)
