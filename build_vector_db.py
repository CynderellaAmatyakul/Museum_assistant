import os
import re
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

TXT_DIR = "database"
DB_SAVE_PATH = "database/vector_store"

# ✅ ฟังก์ชันล้างข้อความภาษาไทย
def clean_text(text: str) -> str:
    text = re.sub(r'([ก-๙])\s(?=[ก-๙่-๋])', r'\1', text)     # พ ิ → พิ
    text = re.sub(r'\s([่-๋])', r'\1', text)                 # า ่ → า่
    text = re.sub(r'([เแโใไ])\s', r'\1', text)              # เ พ → เพ
    text = re.sub(r'\s+', ' ', text)                        # ช่องว่างซ้ำ
    return text.strip()

# ✅ โหลดไฟล์ .txt ทั้งหมดจากโฟลเดอร์
def load_all_txts(txt_dir: str) -> list[Document]:
    documents = []
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            path = os.path.join(txt_dir, filename)
            with open(path, encoding="utf-8") as f:
                raw_text = f.read()
                cleaned = clean_text(raw_text)
                documents.append(Document(
                    page_content=cleaned,
                    metadata={"source": path}
                ))
    return documents

# ✅ Split แล้วทำ FAISS Vector Store
def build_vector_store():
    print("📄 Loading .txt files...")
    raw_documents = load_all_txts(TXT_DIR)

    print("✂️ Splitting texts...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(raw_documents)

    print("🧠 Generating embeddings...")
    embeddings = OpenAIEmbeddings()

    print("📦 Building vector DB with FAISS...")
    vector_db = FAISS.from_documents(split_docs, embeddings)
    vector_db.save_local(DB_SAVE_PATH)
    print("✅ Vector DB saved to:", DB_SAVE_PATH)

if __name__ == "__main__":
    build_vector_store()