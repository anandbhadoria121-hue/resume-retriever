# RAG Resume Retriever

A Retrieval-Augmented Generation system that answers natural-language questions about a
collection of resumes. Ask something like *"Which candidates have UK accounting and IFRS
experience?"* and get a grounded answer with the source resumes cited — backed by semantic
search over a vector database rather than keyword matching.

**Live demo:** https://resume-retriever-3vfslnn5jhk3hhfdhw3l6h.streamlit.app/

> Note: the demo runs on a free tier and sleeps after inactivity — the first request after
> it wakes may take 30–60 seconds.

---

## What it does

Given a job description or a question, the system retrieves the most relevant resume chunks
from a vector index and uses an LLM to generate an answer grounded strictly in that retrieved
context. If the retrieved resumes don't contain the answer, it says so rather than making
something up.

- **Semantic search** over resumes using dense embeddings, not keyword matching
- **Grounded generation** — answers are conditioned on retrieved context and cite source filenames
- **Category filtering** — optionally restrict search to a job category
- **Retrieval inspection** — the UI shows the exact chunks the model saw, with similarity scores

---

## Architecture

```
PDF resumes
    │
    ▼
Parse (PyMuPDF) ──► Clean text ──► Chunk (recursive splitter)
    │
    ▼
Embed with BAAI/bge-small-en-v1.5  (384-dim, normalized)
    │
    ▼
Pinecone vector index (cosine)
    │
    ▼
Query ──► embed query ──► top-k retrieval ──► build context
    │
    ▼
LLM (Gemini, with Llama 3.3 fallback) ──► grounded answer + sources
    │
    ▼
Streamlit UI
```

**Two-stage pipeline.** Retrieval finds the right resumes; generation answers from them. Keeping
these separate makes the system debuggable — you can inspect what was retrieved independently of
what was generated.

---

## Tech stack

| Component        | Choice                                             |
|------------------|----------------------------------------------------|
| Embeddings       | `BAAI/bge-small-en-v1.5` (384-dim, cosine)         |
| Vector DB        | Pinecone (serverless)                              |
| LLM (primary)    | Google Gemini                                      |
| LLM (fallback)   | Llama 3.3 70B via OpenRouter                        |
| PDF parsing      | PyMuPDF                                             |
| Chunking         | LangChain `RecursiveCharacterTextSplitter`         |
| Frontend         | Streamlit                                          |

The **dual-LLM** design lets the app fall back to Llama (OpenRouter) when Gemini is rate-limited
under higher traffic, so the demo stays responsive.

---

## Setup

### 1. Install

```bash
git clone <your-repo-url>
cd resume-rag
pip install -r requirements.txt
```

### 2. Configure keys

Create a `.env` file in the project root (this file is gitignored — never commit it):

```
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

### 3. Build the index

Point the loader at a folder of resumes organized as `data/<CATEGORY>/<resume>.pdf`, then run
the indexing script to parse, chunk, embed, and upsert everything into Pinecone. This is a
one-time step (re-run only when the resume set changes).

```bash
python data_loading.py
```

### 4. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Dataset

Built and tested on the [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
(Kaggle) — ~2,400 resumes in PDF form, organized into job-category folders (ACCOUNTANT, HR,
ENGINEERING, etc.).

---

## Evaluation

Rather than report an unexamined accuracy number, I ran a focused manual evaluation of the
**retrieval** stage — the half where most RAG errors originate.

### Method

I used real job descriptions as queries, retrieved the top 1 to 15 resumes (Users choice) for each, and judged each
result **relevant / not relevant by reading it** — using a strict bar ("does this resume genuinely
look like this role?"). Precision@10 is the fraction of the 10 retrieved that were relevant.

I deliberately did **not** use the dataset's category labels as ground truth (see findings).

### Results

| Query (job description)        | Precision@10 |
|--------------------------------|:------------:|
| Accounting (UK, IFRS/GAAP)     | 1.00         |
| Recruitment / talent acquisition | 1.00       |
| ESG / sustainability finance   | 0.70         |
| Market-risk analyst (VaR/SVaR) | 0.70         |
| Software / coding role         | 0.60         |
| Ultrasound clinical sales      | 0.40         |
| **Mean**                       | **≈ 0.73**   |

Since these evaluation were currently very limited, I do plan to work on this part and make a python file that can much better evaluate the score. By using Gemini, I plan to make the script so that it can verify if top 10 - 15 chunks are even relevant. For now I would propose to use this program on resumes that were directly submitted on a particular job listing rather than across so many different labels. 

### Findings

1. **Retrieval is strong for single-domain queries with dense vocabulary.** Accounting and
   recruitment scored 1.00 — those fields use distinctive, repeated terminology that embeds
   cleanly.

2. **It degrades on narrow, multi-constraint queries.** The ultrasound-sales query (0.40) asked
   for *sales + medical devices + specific imaging modalities*; single-vector retrieval latched
   onto the dominant generic "sales" signal and under-weighted the specialized clinical
   constraints. ESG-finance and market-risk (both 0.70) showed a milder version of the same effect.

3. **Category labels are unreliable as relevance ground truth.** Inspection turned up resumes
   whose folder label contradicts their content — e.g. a full-stack software developer's resume
   filed under `AGRICULTURE`, and a resume labeled `ACCOUNTANT` that is actually a solar sales rep.
   A category-based metric would have *penalized correct retrievals*. This is why evaluation was
   done by reading, not by label.

4. **Two concrete bugs surfaced:**
   - **Duplicate results** — the same resume occasionally appears multiple times in the top-k,
     wasting result slots. Fixable by deduplicating on filename before returning.
   - **Chunking splits skills from dates** — queries about *years of experience* often can't be
     answered because chunking separates the skills section from the dated work history. Better
     handled by extracting `total_years_experience` into metadata at index time than by leaving
     it to semantic search.
---

## Possible improvements

- **Hybrid retrieval** (dense + BM25 keyword) to better handle narrow multi-constraint queries
  where a rare specialized term should carry more weight.
- **Structured metadata** (years of experience, certifications, location) extracted at index
  time, enabling filters that semantic search handles poorly.
- **Deduplication** on filename in the retrieval step.
- **Reranking** the top-k with a cross-encoder for sharper ordering.
- **Automating evaluation task**
- **Ways to add data with some fail safe from the user end**

---

## Project structure

```
resume-rag/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # retrieval + generation logic
├── data_loading.py     # one-time indexing: parse, chunk, embed, upsert
├── requirements.txt
├── .env                # API keys (gitignored)
└── README.md
```
