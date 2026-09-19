# Darukaa.Earth — AI Biodiversity Intelligence Chatbot

An AI system that reasons like an environmental scientist: it retrieves grounded
knowledge about soil, land use, climate, and human-impact factors, connects
multiple variables together, and produces evidence-backed, non-generic
biodiversity recommendations — not a generic LLM wrapper.

## Architecture

```
User (text or JSON) 
      │
      ▼
┌─────────────────────┐
│  Variable Extractor  │  → pulls soil organic carbon, rainfall, land use,
│  (app.py)            │     region from free text or structured JSON input
└─────────┬────────────┘
          │
          ▼
   Enough variables?  ──No──► Ask ONE clarifying question (multi-turn memory
          │                    via st.session_state keeps context across turns)
         Yes
          │
          ▼
┌─────────────────────┐
│   Retrieval (RAG)    │  → Chroma vector store, free local embedding model
│                      │     (all-MiniLM-L6-v2, runs on-device), top-5 chunks
└─────────┬────────────┘
          │
          ▼
┌─────────────────────┐
│  Reasoning (LLM)     │  → Google Gemini 1.5 Flash (free tier), system
│                      │     prompt enforces:
│                      │     - multi-variable causal chains (3+ variables)
│                      │     - fixed output structure (Recommendation / Why /
│                      │       Metrics / Time horizon / Confidence / Source)
│                      │     - no fabrication outside retrieved context
└─────────┬────────────┘
          │
          ▼
     Structured, cited recommendation shown in chat
```

### Why this design
- **Knowledge grounding, not prompt-only reasoning**: all scientific claims
  come from `knowledge_base/*.txt`, chunked by tagged entries (e.g. `[SOIL-02]`)
  and embedded into a vector store. The LLM is instructed to cite the tag(s)
  it used and is told not to invent facts outside retrieved context.
- **Multi-variable reasoning is enforced structurally**, not just requested:
  the system prompt requires every recommendation to connect at least 3
  environmental variables, and the knowledge base itself is written to
  describe cross-variable causal chains (e.g. soil organic carbon → microbial
  diversity → pollinator support), not isolated facts.
- **Conversational intelligence**: the app tracks which of 4 key variables
  (soil organic carbon, rainfall, land use, region) are known at any point in
  the conversation (via regex extraction + optional structured JSON input) and
  asks a targeted clarifying question when information is insufficient —
  matching the spec's example flow.

## Database / Schema

No external database is required for this prototype — the knowledge base is
stored as flat text files, chunked and embedded into an **in-memory Chroma
collection** at runtime (rebuilt once per session, cached via
`st.cache_resource`).

**Knowledge base schema** (per chunk):
| Field | Description |
|---|---|
| `id` | Tag, e.g. `SOIL-02` |
| `document` | Full text of the knowledge snippet, including source citation |
| `metadata.source_file` | Originating `.txt` file (e.g. `soil_health.txt`) |

**Session state schema** (Streamlit, in-memory per user session):
| Field | Description |
|---|---|
| `messages` | Full conversation history (role + content), giving multi-turn memory |
| `known_vars` | Dict of extracted variables (`soil_organic_carbon`, `rainfall`, `land_use`, `region`) accumulated across turns |

Knowledge base coverage: `soil_health.txt`, `land_use.txt`,
`climate_factors.txt`, `human_impact.txt` — 18 total tagged knowledge
snippets covering soil, biodiversity, climate, and human-impact interactions
with cited sources (FAO, IPCC, IPBES, ICRAF, CGIAR).

## Local Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Paste your free Google Gemini API key into the sidebar when the app opens
   in your browser (get one at aistudio.google.com/app/apikey — no credit
   card required). The key is entered at runtime and never stored.

## Deployment (Live Demo)

Deployed via **Streamlit Community Cloud**:
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the GitHub
   repo, and set the main file to `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically on each push —
   this is the project's CI/CD: every commit to `main` triggers an automatic
   rebuild and redeploy of the live app, no separate pipeline configuration
   required.
4. The Gemini API key is entered by the end user in the sidebar at runtime
   (not stored as a server secret), so no credentials are baked into the
   deployment, and no cost is incurred by the demo owner.

## Example Interaction

**Input:** "Biodiversity is declining on my land"
**System:** Asks for soil organic carbon %, rainfall pattern, and land use
type (matching the spec's example).

**Input (once sufficient):** soil organic carbon 0.3%, low rainfall,
monoculture wheat, semi-arid region
**System output:**
- **Recommendation:** Transition from monoculture wheat to agroforestry
  intercropping with drought-tolerant tree species
- **Why it works:** Low SOC (0.3%) combined with low rainfall compounds soil
  degradation; agroforestry root systems increase moisture retention and
  organic matter input simultaneously, breaking the soil-water-biodiversity
  feedback loop
- **Metrics impacted:** Soil organic carbon, soil moisture retention,
  pollinator habitat
- **Time horizon:** Medium to long-term (5-10 years for full effect)
- **Confidence:** High
- **Source:** [SOIL-03], [CLIM-01]

## Known Limitations / Next Steps

- Variable extraction from free text uses regex heuristics; a production
  version would use the LLM itself for structured extraction.
- Knowledge base is currently 18 curated snippets; scaling would mean
  ingesting full FAO/IPCC PDF reports with a proper chunking + citation
  pipeline.
- No persistent database — vector store rebuilds each session. Production
  would use a persistent Chroma/Pinecone instance.
- Geo-coordinate input is not yet implemented (listed as bonus in the spec).
