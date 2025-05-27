import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time
import tempfile
import shutil
import logging
from datetime import datetime
import chromadb
import json
import uuid
import atexit
import hashlib

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from .llm_config import embeddings, llm_general, llm_coding
from .utils import extract_javascript_from_text, convert_js_to_python_code

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent.parent
BASE_CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
PDF_STORAGE_PATH = str(BASE_DIR / "data" / "pdfs")
PDF_METADATA_PATH = str(BASE_DIR / "data" / "pdf_metadata.json")
PDF_INDEX_PATH = str(BASE_DIR / "data" / "pdf_index.json")

# 현재 사용 중인 ChromaDB 경로 (기본값으로 초기화)
CHROMA_DB_PATH = BASE_CHROMA_DB_PATH

# PDF 메타데이터 관리
pdf_metadata = {}
pdf_index = {}  # PDF 파일 경로와 ID 매핑

# 정리할 데이터베이스 목록
databases_to_cleanup = set()

# 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# ChromaDB 초기화
vectorstore = None

def save_pdf_metadata():
    """PDF 메타데이터를 파일에 저장합니다."""
    with open(PDF_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(pdf_metadata, f, ensure_ascii=False, indent=2)

def load_pdf_metadata():
    """PDF 메타데이터를 파일에서 로드합니다."""
    global pdf_metadata
    if os.path.exists(PDF_METADATA_PATH):
        with open(PDF_METADATA_PATH, 'r', encoding='utf-8') as f:
            pdf_metadata = json.load(f)

def save_pdf_index():
    """PDF 인덱스를 파일에 저장합니다."""
    with open(PDF_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(pdf_index, f, ensure_ascii=False, indent=2)

def load_pdf_index():
    """PDF 인덱스를 파일에서 로드합니다."""
    global pdf_index
    if os.path.exists(PDF_INDEX_PATH):
        with open(PDF_INDEX_PATH, 'r', encoding='utf-8') as f:
            pdf_index = json.load(f)

# 메타데이터와 인덱스 로드
load_pdf_metadata()
load_pdf_index()

def get_new_db_path():
    """Generate a new unique database path."""
    return f"{BASE_CHROMA_DB_PATH}_{uuid.uuid4().hex[:8]}"

def cleanup_database(db_path):
    """Clean up a single database directory."""
    if not os.path.exists(db_path):
        return
    
    try:
        # 먼저 SQLite 파일을 삭제
        sqlite_file = os.path.join(db_path, "chroma.sqlite3")
        if os.path.exists(sqlite_file):
            try:
                os.remove(sqlite_file)
            except Exception as e:
                print(f"Warning: Could not remove SQLite file {sqlite_file}: {e}")
                return False
        
        # 나머지 파일들 삭제
        for root, dirs, files in os.walk(db_path, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception as e:
                    print(f"Warning: Could not remove file {name}: {e}")
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception as e:
                    print(f"Warning: Could not remove directory {name}: {e}")
        
        # 마지막으로 디렉토리 삭제
        try:
            os.rmdir(db_path)
            return True
        except Exception as e:
            print(f"Warning: Could not remove directory {db_path}: {e}")
            return False
    except Exception as e:
        print(f"Error cleaning up database {db_path}: {e}")
        return False

def cleanup_old_databases():
    """Clean up old database directories."""
    global databases_to_cleanup
    
    for db_path in list(databases_to_cleanup):
        if cleanup_database(db_path):
            databases_to_cleanup.remove(db_path)

def register_cleanup(db_path):
    """Register a database for cleanup."""
    global databases_to_cleanup
    databases_to_cleanup.add(db_path)

# 프로그램 종료 시 정리 작업 등록
atexit.register(cleanup_old_databases)

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_duplicate_file(file_path: str) -> Optional[str]:
    """Check if file is a duplicate based on content hash."""
    file_hash = calculate_file_hash(file_path)
    for pdf_id, info in pdf_index.items():
        if os.path.exists(info["path"]):
            existing_hash = calculate_file_hash(info["path"])
            if existing_hash == file_hash:
                return pdf_id
    return None

def update_pdf_status(pdf_id: str, status: str, error_message: Optional[str] = None):
    """Update PDF processing status and save metadata."""
    if pdf_id in pdf_metadata:
        pdf_metadata[pdf_id]["status"] = status
        if error_message:
            pdf_metadata[pdf_id]["error"] = error_message
        pdf_metadata[pdf_id]["last_updated"] = datetime.now().isoformat()
        save_pdf_metadata()

def retry_failed_pdfs():
    """Retry processing of failed PDFs."""
    for pdf_id, info in pdf_metadata.items():
        if info["status"] == "failed" and pdf_id in pdf_index:
            pdf_path = pdf_index[pdf_id]["path"]
            if os.path.exists(pdf_path):
                print(f"Retrying failed PDF: {pdf_path}")
                update_pdf_status(pdf_id, "processing")
                if process_and_embed_pdf(pdf_path):
                    update_pdf_status(pdf_id, "processed")
                else:
                    update_pdf_status(pdf_id, "failed", "Retry failed")

def process_and_embed_pdf(pdf_file_path: str) -> bool:
    """
    PDF 파일을 처리하고, JavaScript를 Python으로 변환 후 Vector DB에 저장합니다.
    성공 시 True, 실패 시 False 반환.
    """
    try:
        print(f"Processing PDF: {pdf_file_path}")
        
        # 중복 파일 체크
        duplicate_id = is_duplicate_file(pdf_file_path)
        if duplicate_id:
            print(f"Duplicate file detected. Reusing existing data for {pdf_file_path}")
            update_pdf_status(duplicate_id, "processed")
            return True

        # PDF ID 찾기
        pdf_id = None
        for pid, info in pdf_index.items():
            if info["path"] == pdf_file_path:
                pdf_id = pid
                break

        if pdf_id:
            update_pdf_status(pdf_id, "processing")

        # PyPDFLoader 사용 (더 안정적인 PDF 처리)
        loader = PyPDFLoader(pdf_file_path)
        
        docs = loader.load()
        
        processed_docs: List[Document] = []
        for doc in docs:
            content = doc.page_content
            js_codes = extract_javascript_from_text(content)
            
            if js_codes:
                print(f"Found {len(js_codes)} JavaScript blocks in a chunk. Converting to Python...")
                for js_code in js_codes:
                    python_code = convert_js_to_python_code(js_code, llm_coding)
                    content = content.replace(js_code, f"\n'''\nOriginal JavaScript:\n{js_code}\n'''\n\n'''\nConverted Python:\n{python_code}\n'''\n")
                
                doc.page_content = content

            processed_docs.append(doc)

        if not processed_docs:
            error_msg = f"No text could be extracted from {pdf_file_path}"
            if pdf_id:
                update_pdf_status(pdf_id, "failed", error_msg)
            return False

        split_docs = text_splitter.split_documents(processed_docs)
        
        if not split_docs:
            error_msg = f"No text chunks generated after splitting for {pdf_file_path}"
            if pdf_id:
                update_pdf_status(pdf_id, "failed", error_msg)
            return False
            
        vectorstore.add_documents(split_docs)
        vectorstore.persist()
        
        if pdf_id:
            update_pdf_status(pdf_id, "processed")
        
        return True
        
    except Exception as e:
        error_msg = f"Error processing PDF {pdf_file_path}: {str(e)}"
        logger.error(error_msg)
        if pdf_id:
            update_pdf_status(pdf_id, "failed", error_msg)
        return False

def get_relevant_documents(query: str, k: int = 5) -> List[Document]:
    """Vector DB에서 관련 문서를 검색합니다."""
    global vectorstore
    try:
        if vectorstore is None:
            vectorstore = initialize_chroma(force_recreate=False)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)
        return relevant_docs
    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        # 오류 발생 시 DB 재초기화 시도
        vectorstore = initialize_chroma(force_recreate=True)
        return []

def query_pdf_content(query: str, k: int = 5) -> str:
    """PDF 내용을 검색하고 관련 내용을 반환합니다."""
    try:
        docs = get_relevant_documents(query, k)
        if not docs:
            return "관련된 PDF 내용을 찾을 수 없습니다."
        
        # 관련 문서 내용을 하나의 문자열로 결합
        content = "\n\n".join([doc.page_content for doc in docs])
        return content
    except Exception as e:
        print(f"Error querying PDF content: {e}")
        return "PDF 내용을 검색하는 중 오류가 발생했습니다."

def get_rag_retriever(k: int = 5):
    """RAG 검색기 인스턴스를 반환합니다."""
    return vectorstore.as_retriever(search_kwargs={"k": k})

def list_available_collections():
    """ 사용 가능한 ChromaDB 컬렉션 목록 (디버깅용) """
    client = Chroma(persist_directory=CHROMA_DB_PATH)
    chromadb_client = client._client
    collection = chromadb_client.list_collections()
    return [col.name for col in collection]

# 초기화 시 기존 DB 로드 확인
print(f"ChromaDB initialized. Available collections: {list_available_collections()}")
if not any(col == "rag_collection" for col in list_available_collections()):
    print("Warning: 'rag_collection' not found. PDFs might need to be re-uploaded.")

def get_processed_pdfs() -> List[Dict]:
    """
    처리된 PDF 파일 목록을 반환합니다.
    """
    return [
        {
            "filename": info["filename"],
            "id": info["id"],
            "status": pdf_metadata[info["id"]]["status"]
        }
        for info in pdf_index.values()
    ]

def initialize_chroma(force_recreate: bool = False) -> Chroma:
    """Initialize or load existing ChromaDB."""
    global CHROMA_DB_PATH, vectorstore
    
    if not force_recreate and os.path.exists(BASE_CHROMA_DB_PATH):
        CHROMA_DB_PATH = BASE_CHROMA_DB_PATH
        print(f"Using existing ChromaDB at {CHROMA_DB_PATH}")
    else:
        CHROMA_DB_PATH = get_new_db_path()
        print(f"Creating new ChromaDB at {CHROMA_DB_PATH}")
    
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name="rag_collection"
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        # 오류 발생 시 새로운 DB 생성
        CHROMA_DB_PATH = get_new_db_path()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name="rag_collection"
        )
        return vectorstore

# ChromaDB 초기화 실행
vectorstore = initialize_chroma(force_recreate=False)

# 주기적으로 실패한 PDF 재처리 시도
def start_pdf_retry_scheduler():
    """Start a background thread to periodically retry failed PDFs."""
    import threading
    import time

    def retry_worker():
        while True:
            retry_failed_pdfs()
            time.sleep(300)  # 5분마다 재시도

    thread = threading.Thread(target=retry_worker, daemon=True)
    thread.start()

# 초기화 시 실패한 PDF 재처리 시작
start_pdf_retry_scheduler()