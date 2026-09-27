# check_db.py
import db_patch  # Must be line 1 to apply sqlite3 override

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

total_chunks = vectorstore._collection.count()
raw_data = vectorstore.get(include=["metadatas"])
metadatas = raw_data.get("metadatas", [])

unique_files = sorted(list({
    m.get("filename") or m.get("source") or "Unknown" 
    for m in metadatas if m
}))

print("==========================================")
print(f"TOTAL INDEXED CHUNKS IN CHROMADB : {total_chunks}")
print(f"TOTAL UNIQUE PDF FILES           : {len(unique_files)}")
print("==========================================")
print("List of Ingested PDFs:")
for idx, filename in enumerate(unique_files, start=1):
    print(f"  {idx}. {filename}")
print("==========================================")