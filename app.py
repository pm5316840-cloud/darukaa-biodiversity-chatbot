"""
Darukaa.Earth — AI Biodiversity Intelligence Chatbot
A RAG-based conversational system that reasons across soil, land use,
climate, and human-impact variables to produce evidence-backed
biodiversity recommendations.

Architecture:
  1. Knowledge base (knowledge_base/*.txt) is chunked and embedded into
     a Chroma vector store (built once, cached).
  2. User input (free text or structured JSON) is parsed to extract known
     environmental variables (soil organic carbon, rainfall, land use, region).
  3. If fewer than 3 key variables are present, the system asks a
     clarifying question instead of answering (per spec requirement).
  4. Once enough context exists, relevant knowledge chunks are retrieved
     and passed to the LLM with a strict output-format prompt that forces
     multi-variable reasoning and citation of the retrieved source.
  5. Conversation history is kept in Streamlit session state, giving the
     system multi-turn memory.
"""

import os
import re
import json
import time
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Uses Google Gemini's free API tier (no credit card required) for reasoning,
# and Chroma's built-in local embedding model (runs on-device, no API key,
# no cost) for retrieval — so the whole app is free to run.
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
COLLECTION_NAME = "darukaa_kb"
REQUIRED_VARS = ["soil_organic_carbon", "rainfall", "land_use", "region"]
CHAT_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT = """You are an AI environmental scientist for Darukaa.Earth. You reason
about biodiversity, soil, climate, and land-use interactions using ONLY the
retrieved knowledge snippets provided to you as context. Do not invent facts,
figures, or sources outside the provided context.

For every recommendation you give, you MUST connect at least 3 distinct
environmental variables (e.g. soil organic carbon, rainfall, land use,
species richness, microbial activity) — never give a single-variable answer.

Respond in this exact structure for each recommendation:

**Recommendation:** <specific action, not generic advice>
**Why it works:** <scientific reasoning tracing the causal chain across variables>
**Metrics impacted:** <list the specific metrics, e.g. soil organic carbon, pollinator abundance>
**Time horizon:** <short-term / medium-term / long-term, with approximate duration>
**Confidence:** <high / medium / low, based on how directly the retrieved context supports this>
**Source:** <cite the [CODE-NN] tag(s) from the retrieved context you used>

Never say generic things like "use sustainable practices." Every claim must be
traceable to the provided context. If the context doesn't support a strong
recommendation, say so honestly rather than fabricating one.
"""

CLARIFYING_PROMPT_TEMPLATE = """You are an AI environmental scientist for Darukaa.Earth.
The user has given incomplete information. Based on what they HAVE told you,
ask ONE focused clarifying question to get the most valuable missing variable
(soil organic carbon %, rainfall pattern, land use / crop type, or region/climate zone).
Be specific and brief — do not lecture, just ask.

What the user has told you so far:
{known}

Missing:
{missing}
"""

# --------------------------------------------------------------------------
# Vector store setup (cached across reruns)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing knowledge base...")
def get_collection():
    client = chromadb.EphemeralClient()
    # Free, local embedding model (all-MiniLM-L6-v2) — downloads once from
    # Hugging Face on first run, then runs entirely on-device. No API key,
    # no cost.
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    ids, docs, metas = [], [], []
    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(KB_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Split on entries like [SOIL-01] ... up to the next tag or EOF
        chunks = re.split(r"(?=\[[A-Z]+-\d+\])", content)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            match = re.match(r"\[([A-Z]+-\d+)\]", chunk)
            chunk_id = match.group(1) if match else f"{fname}-{len(ids)}"
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"source_file": fname})

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    return collection


def retrieve_context(collection, query: str, n_results: int = 5):
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0] if results["documents"] else []
    return "\n\n---\n\n".join(docs)


def safe_generate(model, content, temperature):
    """Calls Gemini and returns friendly text instead of crashing on rate limits."""
    try:
        response = model.generate_content(
            content, generation_config={"temperature": temperature}
        )
        return response.text
    except ResourceExhausted:
        return (
            "⏳ **Free tier rate limit reached.** This demo uses Google Gemini's "
            "free API tier, which allows a limited number of requests per minute. "
            "Please wait about 60 seconds and send your message again."
        )
    except GoogleAPICallError as e:
        return f"⚠️ The AI service returned an error: {e}. Please try again in a moment."


# --------------------------------------------------------------------------
# Input parsing — free text + structured JSON
# --------------------------------------------------------------------------
def try_parse_json(text: str):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def extract_known_vars(all_user_text: str, structured: dict):
    """Best-effort extraction of key variables from conversation so far."""
    known = dict(structured) if structured else {}
    text = all_user_text.lower()

    if "soil_organic_carbon" not in known:
        m = re.search(r"(soil organic carbon|soc)[^\d]{0,15}([\d.]+)\s*%?", text)
        if m:
            known["soil_organic_carbon"] = m.group(2) + "%"

    if "rainfall" not in known:
        for kw in ["low rainfall", "high rainfall", "moderate rainfall", "semi-arid", "arid"]:
            if kw in text:
                known["rainfall"] = kw
                break

    if "land_use" not in known:
        for kw in ["monoculture", "intercropping", "agroforestry", "polyculture", "wheat", "maize", "rice"]:
            if kw in text:
                known["land_use"] = kw
                break

    if "region" not in known:
        m = re.search(r"(semi-arid|arid|tropical|temperate|subtropical)\s*region", text)
        if m:
            known["region"] = m.group(1)

    return known


def missing_vars(known: dict):
    return [v for v in REQUIRED_VARS if v not in known]


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Darukaa.Earth — Biodiversity AI", page_icon="🌱", layout="centered")
st.title("🌱 Darukaa.Earth — AI Biodiversity Intelligence")
st.caption("An AI environmental scientist: evidence-backed, multi-variable biodiversity recommendations.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Free — get one at aistudio.google.com/app/apikey (no credit card required).",
    )
    st.divider()
    st.subheader("Structured input (optional)")
    st.caption("Paste JSON instead of / in addition to typing below.")
    json_input = st.text_area(
        "e.g. {\"soil_organic_carbon\": \"0.3%\", \"rainfall\": \"low\", \"land_use\": \"monoculture wheat\", \"region\": \"semi-arid\"}",
        height=120,
    )
    st.divider()
    if st.button("Reset conversation"):
        st.session_state.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "known_vars" not in st.session_state:
    st.session_state.known_vars = {}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not api_key:
    st.info("Enter your free Gemini API key in the sidebar to start.")
    st.stop()

collection = get_collection()
genai.configure(api_key=api_key)
model = genai.GenerativeModel(CHAT_MODEL)

user_input = st.chat_input("Describe your land, e.g. 'Biodiversity is declining on my farm'")

if user_input:
    structured = try_parse_json(json_input) if json_input.strip() else None

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    all_text = " ".join(m["content"] for m in st.session_state.messages if m["role"] == "user")
    st.session_state.known_vars = extract_known_vars(all_text, structured)
    missing = missing_vars(st.session_state.known_vars)

    with st.chat_message("assistant"):
        if len(missing) >= 2:
            # Not enough signal yet — ask a clarifying question
            prompt = CLARIFYING_PROMPT_TEMPLATE.format(
                known=json.dumps(st.session_state.known_vars, indent=2) or "Nothing yet",
                missing=", ".join(missing),
            )
            reply = safe_generate(model, prompt, 0.3)
        else:
            # Enough context — retrieve and reason
            context = retrieve_context(collection, all_text)
            full_prompt = f"""Known environmental variables:
{json.dumps(st.session_state.known_vars, indent=2)}

Retrieved knowledge context:
{context}

User's latest message: {user_input}

Provide your recommendation(s) following the required output structure."""
            reply = safe_generate(model, [SYSTEM_PROMPT, full_prompt], 0.4)

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

with st.expander("Known variables extracted so far"):
    st.json(st.session_state.known_vars)
