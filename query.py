import db_patch  # Must remain at line 1

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

os.environ["ANONYMIZED_TELEMETRY"] = "False"
DB_DIR = "./chroma_db"

def main():
    print("Loading ChromaDB vectorstore and FastEmbed model (this takes a few seconds)...")
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    print("Ready!\n" + "="*50)

    while True:
        try:
            user_query = input("\nAsk a question about MINAGRI policies (or type 'exit' / 'q' to quit):\n> ").strip()
            
            if not user_query:
                continue
                
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting search. Good job!")
                break

            results = vectorstore.similarity_search_with_score(user_query, k=3)
            
            print(f"\nTop 3 Relevant Chunks for: '{user_query}'\n" + "-"*50)
            for idx, (doc, score) in enumerate(results, 1):
                filename = doc.metadata.get("filename", "Unknown")
                print(f"\n--- Result {idx} | Source: {filename} (Distance Score: {score:.4f}) ---")
                print(f"Content:\n{doc.page_content}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting search.")
            break

if __name__ == "__main__":
    main()