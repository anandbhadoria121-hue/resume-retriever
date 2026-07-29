import os
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from openai import OpenAI
import configparser

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about resumes. "
    "Answer ONLY using the provided context. If the context does not "
    "contain the answer, say you don't know. Cite the filename of the "
    "resume(s) you used."
)


class RAGPipeline:
    def __init__(self, index_name="resumes"):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        self.index = Pinecone(api_key=api_key).Index(index_name)

        self.embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Gemini client (GEMINI_API_KEY env var)
        gemini_key = os.getenv("GOOGLE_API_KEY")
        self.gemini = genai.Client(api_key=gemini_key) if gemini_key else None

        or_key = os.getenv("GROQ_API_KEY")
        self.openrouter = (
            OpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key)
            if or_key
            else None
        )



    def retrieve(self, query, top_k=5, job_class=None):
        """Embed the query and fetch the top_k most similar chunks."""
        query_vec = self.embed_model.embed_query(BGE_QUERY_PREFIX + query)

        filter_ = {"class": {"$eq": job_class}} if job_class else None

        result = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            filter=filter_,
        )
        return result["matches"]

    @staticmethod
    def build_context(matches):
        """Format retrieved chunks into a context block for the LLM."""
        parts = []
        for m in matches:
            meta = m["metadata"]
            parts.append(
                f"[source: {meta['filename']} | class: {meta['class']} "
                f"| score: {m['score']:.3f}]\n{meta['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ---------- GENERATION ----------

    def generate_gemini(self, query, context, model="gemini-3.6-flash"):
        if self.gemini is None:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        response = self.gemini.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text

    def generate_llama(self, query, context,
                       model="meta-llama/llama-3.3-70b-instruct"):
        if self.openrouter is None:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        response = self.openrouter.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                dict(role="user", content=f"Context:\n{context}\n\nQuestion: {query}"),
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    # ---------- FULL PIPELINE ----------

    def ask(self, query, llm="gemini", top_k=5, job_class=None):
        """Retrieve -> build context -> generate. llm: 'gemini' or 'llama'."""
        matches = self.retrieve(query, top_k=top_k, job_class=job_class)
        if not matches:
            return "No relevant chunks found in the index."

        context = self.build_context(matches)

        if llm == "gemini":
            answer = self.generate_gemini(query, context)
        elif llm == "llama":
            answer = self.generate_llama(query, context)
        else:
            raise ValueError("llm must be 'gemini' or 'llama'")

        sources = sorted({m["metadata"]["filename"] for m in matches})
        return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    rag = RAGPipeline(index_name="resumes")

    result = rag.ask(
        "Which candidates have experience with financial reporting "
        "and account reconciliation?",
        llm="gemini",          # or "llama"
        top_k=5,
        # job_class="ACCOUNTANT",  # optional metadata filter
    )
    print(result["answer"])
    print("\nSources:", result["sources"])