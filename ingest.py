import db_patch  # Must remain at line 1

import os
import io
import shutil
import gc
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pypdf import PdfReader

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

os.environ["ANONYMIZED_TELEMETRY"] = "False"
DB_DIR = "./chroma_db"

def clean_old_db():
    if os.path.exists(DB_DIR):
        print("Cleaning old vector database...")
        shutil.rmtree(DB_DIR)

def get_pdf_urls_from_page(page_url: str) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(page_url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    
    pdf_urls = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.lower().endswith(".pdf") or ".pdf?" in href.lower():
            full_url = urljoin(page_url, href)
            pdf_urls.append(full_url)
            
    return list(set(pdf_urls))

def extract_text_from_pdf(pdf_url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    res = requests.get(pdf_url, headers=headers, timeout=30)
    res.raise_for_status()
    
    pdf_file = io.BytesIO(res.content)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def process_and_store_minagri(pages: list, micro_batch_size: int = 100):
    clean_old_db()
    
    # Configure FastEmbed to use small internal batch sizes (32 docs at a time in memory)
    embeddings = FastEmbedEmbeddings(batch_size=32)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Collect all PDF URLs across pages
    all_pdf_urls = []
    for page_url in pages:
        print(f"Finding PDF links on: {page_url}")
        urls = get_pdf_urls_from_page(page_url)
        all_pdf_urls.extend(urls)
        
    all_pdf_urls = list(set(all_pdf_urls))
    print(f"\nFound {len(all_pdf_urls)} unique PDF documents to process.")
    
    total_chunks_processed = 0
    
    for idx, pdf_url in enumerate(all_pdf_urls, 1):
        filename = pdf_url.split("/")[-1].split("?")[0]
        print(f"\n[{idx}/{len(all_pdf_urls)}] Processing: {filename}")
        
        try:
            pdf_text = extract_text_from_pdf(pdf_url)
            if len(pdf_text.strip()) <= 100:
                print(f"  Skipped (empty or scanned PDF)")
                continue
                
            doc = Document(page_content=pdf_text, metadata={"source": pdf_url, "filename": filename})
            chunks = text_splitter.split_documents([doc])
            
            print(f"  Extracted {len(chunks)} chunks. Writing to ChromaDB...")
            
            # Stream chunks into ChromaDB in micro-batches
            for i in range(0, len(chunks), micro_batch_size):
                batch = chunks[i : i + micro_batch_size]
                vectorstore.add_documents(documents=batch)
                
            total_chunks_processed += len(chunks)
            print(f"  Successfully indexed {filename} (Total Chunks so far: {total_chunks_processed})")
            
            # Free memory immediately after each PDF
            del pdf_text
            del doc
            del chunks
            gc.collect()
            
        except Exception as e:
            print(f"  Failed to process {filename}: {e}")

    print(f"\nIngestion complete! Successfully indexed {total_chunks_processed} chunks into ChromaDB.")

if __name__ == "__main__":
    publication_pages = [
        "https://www.minagri.gov.rw/publications/policies-and-strategies",
        "https://www.minagri.gov.rw/publications/policies-and-strategies?tx_filelist_filelist%5Bcontroller%5D=File&tx_filelist_filelist%5BcurrentPage%5D=2&tx_filelist_filelist%5Bpath%5D=&cHash=2c704f94496e3f6e56d862aaa1959888"
    ]
    process_and_store_minagri(publication_pages)