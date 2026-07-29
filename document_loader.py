import os
import re
import time
import torch
import unicodedata
import pandas as pd
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# bge-small is ~5x smaller than bge-m3 and far faster on CPU.
# Switch back to "BAAI/bge-m3" + 1024 only if you have a working GPU.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIMENSION = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean_text(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", t)                 # fix hyphen-split words
    t = re.sub(r"\bCity\s*,\s*State\s*,?\s*(USA)?\b", "", t)  # anonymizer placeholders
    t = re.sub(r"\bCompany Name\b", "", t)
    t = re.sub(r"\s+([,.;:])", r"\1", t)                      # space before punctuation
    t = re.sub(r"[ \t]+", " ", t)                             # collapse spaces/tabs
    t = re.sub(r"\n\s*\n+", "\n\n", t)                        # collapse blank lines
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)                    # join single line-breaks
    return t.strip()


class DataLoading:
    def __init__(self, file_path, csv_path="extracted_resumes.csv"):
        self.file_path = Path(file_path)
        self.csv_path = csv_path
        self._embed_model = None  # lazy-loaded, used only at upload time now

    @property
    def embed_model(self):
        if self._embed_model is None:
            print(f"Loading embedding model '{EMBED_MODEL_NAME}' on {DEVICE}...")
            self._embed_model = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"device": DEVICE},
                encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
            )
        return self._embed_model

    def load(self):
        data = []

        for class_folder in self.file_path.iterdir():
            if not class_folder.is_dir():
                continue
            class_name = class_folder.name

            for pdf_path in class_folder.glob("*.pdf"):
                try:
                    pages = PyMuPDFLoader(str(pdf_path)).load()
                    full_text = clean_text(
                        "\n".join(p.page_content for p in pages)
                    )
                    if not full_text:
                        print(f"Skipped empty file: {pdf_path.name}")
                        continue

                    data.append({
                        "data": full_text,
                        "class": class_name,
                        "filename": pdf_path.name,
                    })
                except Exception as e:
                    print(f"Failed to load {pdf_path.name}: {e}")

        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        print(f"Loaded {len(df)} resumes -> {self.csv_path}")
        return df

    def chunking(self, df=None):
        # Use the DataFrame passed in; fall back to the CSV if called standalone
        if df is None:
            df = pd.read_csv(self.csv_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        for doc_id, row in df.iterrows():
            for i, chunk in enumerate(splitter.split_text(str(row["data"]))):
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "data": chunk,
                    "class": row["class"],
                    "filename": row["filename"],
                })

        chunks_df = pd.DataFrame(chunks)
        print(f"Created {len(chunks_df)} chunks from {len(df)} resumes")
        return chunks_df

    def upload_to_pinecone(self, chunks_df, index_name="resumes", recreate_index=False):
        if chunks_df.empty:
            raise ValueError("chunks_df is empty - nothing to upload")

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Set it as an environment variable "
                "(never hardcode the key in source code)."
            )
        pc = Pinecone(api_key=api_key)

        if recreate_index and pc.has_index(index_name):
            print(f"Deleting existing index '{index_name}' to recreate at "
                  f"{EMBED_DIMENSION} dims...")
            pc.delete_index(index_name)
            while pc.has_index(index_name):
                time.sleep(1)

        if not pc.has_index(index_name):
            print(f"Creating index '{index_name}' ({EMBED_DIMENSION} dims)...")
            pc.create_index(
                name=index_name,
                dimension=EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        else:
            existing_dim = pc.describe_index(index_name).dimension
            if existing_dim != EMBED_DIMENSION:
                raise ValueError(
                    f"Index '{index_name}' exists with dimension {existing_dim}, "
                    f"but the model produces {EMBED_DIMENSION}-dim vectors. "
                    f"Re-run with recreate_index=True to rebuild it."
                )

        index = pc.Index(index_name)

        texts = chunks_df["data"].tolist()
        print(f"Embedding {len(texts)} chunks (this is the slow step)...")
        t0 = time.time()
        embeddings = self.embed_model.embed_documents(texts)
        print(f"Embedded in {time.time() - t0:.1f}s")

        vectors = []
        for (_, row), emb in zip(chunks_df.iterrows(), embeddings):
            vectors.append({
                "id": f"{row['doc_id']}-{row['chunk_index']}",
                "values": emb,
                "metadata": {
                    "text": row["data"],
                    "class": row["class"],
                    "filename": row["filename"],
                    "doc_id": int(row["doc_id"]),
                    "chunk_index": int(row["chunk_index"]),
                },
            })

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i:i + batch_size])
            print(f"  upserted {min(i + batch_size, len(vectors))}/{len(vectors)}")

        print(f"Upserted {len(vectors)} vectors to index '{index_name}'")
        return index


if __name__ == "__main__":
    loader = DataLoading(r"C:\Users\anand\PycharmProjects\PythonProject1\docs\data\data")
    df = loader.load()
    chunks = loader.chunking(df)
    loader.upload_to_pinecone(chunks, recreate_index=True)