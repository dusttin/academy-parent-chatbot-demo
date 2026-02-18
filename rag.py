import os
import pypdf
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="academy_qa")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def index_pdf(pdf_path: str, source_name: str = "academy") -> int:
    """학원 안내문 PDF를 ChromaDB에 인덱싱"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        collection.upsert(
            ids=[f"{source_name}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": source_name, "chunk_index": i}]
        )
    return len(chunks)


def index_text(text: str, source_name: str = "faq") -> int:
    """텍스트를 ChromaDB에 인덱싱"""
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        collection.upsert(
            ids=[f"{source_name}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": source_name, "chunk_index": i}]
        )
    return len(chunks)


def query(question: str, top_k: int = 3) -> str:
    """학부모 질문에 대한 RAG 답변 생성"""
    if collection.count() == 0:
        return "아직 학원 안내 자료가 등록되지 않았습니다."

    question_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
    )

    if not results["documents"][0]:
        return "죄송합니다, 관련 정보를 찾지 못했습니다. 원장님께 직접 문의해드릴게요."

    context = "\n\n".join(results["documents"][0])

    system_prompt = (
        "당신은 학원 학부모의 문의에 친절하고 따뜻하게 답변하는 AI 어시스턴트입니다.\n"
        "주어진 학원 안내 자료를 바탕으로 정확하게 답변하세요.\n"
        "자료에 없는 내용은 '원장님께 직접 문의해드릴게요'라고 안내하세요.\n"
        "답변은 학부모가 이해하기 쉽게 친근하고 명확하게 작성하세요."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"학원 안내 자료:\n{context}\n\n학부모 문의: {question}"}
        ],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content


def get_collection_count() -> int:
    return collection.count()
