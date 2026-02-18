import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import RateLimitError, APIError
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import index_pdf, index_text, query, get_collection_count

load_dotenv()

app = FastAPI(title="학원 학부모 응대봇 데모")

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=503,
        content={"detail": "OpenAI API 크레딧이 소진되었습니다. 관리자에게 문의해주세요."}
    )

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=502,
        content={"detail": f"OpenAI API 오류가 발생했습니다: {exc.message}"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    question: str


class PreloadRequest(BaseModel):
    content: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """학원 안내문 PDF 업로드 및 ChromaDB 인덱싱"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunk_count = index_pdf(str(file_path), source_name=file.filename)
    return {
        "message": f"'{file.filename}' 업로드 완료",
        "chunks_indexed": chunk_count,
        "total_indexed": get_collection_count()
    }


@app.post("/preload")
async def preload_sample(request: PreloadRequest):
    """샘플 학원 FAQ 사전 로드 (데모용)"""
    chunk_count = index_text(request.content, source_name="sample_faq")
    return {
        "message": "샘플 데이터 로드 완료",
        "chunks_indexed": chunk_count
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """학부모 문의에 RAG 답변"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="문의 내용을 입력해주세요")

    answer = query(request.question)
    return {"answer": answer}


@app.get("/status")
async def status():
    return {
        "status": "운영 중",
        "indexed_chunks": get_collection_count()
    }
