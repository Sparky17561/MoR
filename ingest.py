"""
Ingest Bitext Customer Support dataset, chunk text, embed and store in ChromaDB.
Also write chunks to disk (chunks.txt) and a word-level vocab file (vocab.txt)
so we can later train MoR on the same text.
"""

import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# -------- CONFIG ----------
CHROMA_DB_DIR = "chromadb_store"
COLLECTION_NAME = "bitext_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 120
CHUNK_OVERLAP = 20
BATCH_EMBED = 256
OUT_CHUNKS_FILE = "data/chunks.txt"
OUT_VOCAB_FILE = "data/vocab.txt"

# Test mode - set to True to run a small test first
TEST_MODE = False
TEST_SAMPLES = 10  # Only process first 10 samples in test mode

os.makedirs("data", exist_ok=True)

def test_setup():
    """Test basic setup before full processing"""
    print("=" * 50)
    print("TESTING SETUP...")
    print("=" * 50)
    
    # Test ChromaDB
    try:
        print(f"Testing ChromaDB at {CHROMA_DB_DIR}...")
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Test collection creation/access
        test_collection_name = "test_collection"
        if test_collection_name in [c.name for c in client.list_collections()]:
            test_collection = client.get_collection(test_collection_name)
            client.delete_collection(test_collection_name)  # Clean up
        
        test_collection = client.create_collection(test_collection_name)
        print(f"✓ ChromaDB working - created test collection")
        
        # Test embedding
        print(f"Testing embeddings with {EMBED_MODEL}...")
        embedder = SentenceTransformer(EMBED_MODEL)
        test_texts = ["This is a test.", "Another test sentence."]
        embeddings = embedder.encode(test_texts, convert_to_numpy=True)
        print(f"✓ Embeddings working - shape: {embeddings.shape}")
        
        # Test adding to collection
        test_collection.add(
            documents=test_texts,
            metadatas=[{"test": True}, {"test": True}],
            ids=["test1", "test2"],
            embeddings=embeddings.tolist()
        )
        count = test_collection.count()
        print(f"✓ Collection operations working - added {count} items")
        
        # Clean up test collection
        client.delete_collection(test_collection_name)
        print("✓ Test cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Setup test failed: {e}")
        return False

def chunk_text(text, chunk_size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Helper: chunk text into overlapping windows of words"""
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

def main():
    # Run setup test first
    if not test_setup():
        print("Setup test failed. Please fix issues before proceeding.")
        return
    
    print("\n" + "=" * 50)
    print("STARTING MAIN PROCESSING...")
    print("=" * 50)
    
    # -------- Chroma client --------
    print(f"Initializing Chroma persistent client at {CHROMA_DB_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection {COLLECTION_NAME} exists. Deleting to start fresh...")
        client.delete_collection(COLLECTION_NAME)
    
    collection = client.create_collection(COLLECTION_NAME)
    print(f"Created new collection: {COLLECTION_NAME}")

    # -------- load dataset --------
    print("Loading bitext/Bitext-customer-support-llm-chatbot-training-dataset...")
    try:
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
        print(f"Dataset loaded successfully. Total samples: {len(ds)}")
        
        if TEST_MODE:
            ds = ds.select(range(min(TEST_SAMPLES, len(ds))))
            print(f"TEST MODE: Processing only first {len(ds)} samples")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # -------- embedder --------
    print(f"Loading embedder: {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # ingest and accumulate texts for vocab/training
    all_chunks = []
    batch_texts, batch_ids, batch_meta = [], [], []

    print("Chunking & ingesting dataset...")

    try:
        for i, example in enumerate(tqdm(ds, desc="Processing rows")):
            instruction = example.get("instruction") or example.get("question") or ""
            response = example.get("response") or ""
            category = example.get("category") or ""
            intent = example.get("intent") or ""
            flags = example.get("flags") or ""

            # Combine instruction + response as a single text chunk
            text = f"Q: {instruction}\nA: {response}"

            chunks = chunk_text(text)
            for ci, c in enumerate(chunks):
                doc_id = f"{i}__{ci}"
                meta = {
                    "source": "bitext_dataset",
                    "category": category,
                    "intent": intent,
                    "flags": flags,
                    "row_idx": i,
                    "chunk_idx": ci
                }
                batch_texts.append(c)
                batch_ids.append(doc_id)
                batch_meta.append(meta)
                all_chunks.append(c)

            # batch embed
            if len(batch_texts) >= BATCH_EMBED:
                print(f"Embedding batch of {len(batch_texts)} texts...")
                embs = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                collection.add(
                    documents=batch_texts, 
                    metadatas=batch_meta, 
                    ids=batch_ids, 
                    embeddings=embs.tolist()
                )
                batch_texts, batch_ids, batch_meta = [], [], []

        # flush remainder
        if batch_texts:
            print(f"Embedding final batch of {len(batch_texts)} texts...")
            embs = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            collection.add(
                documents=batch_texts, 
                metadatas=batch_meta, 
                ids=batch_ids, 
                embeddings=embs.tolist()
            )

        # ChromaDB with PersistentClient automatically persists data
        # No need to call persist() - data is automatically saved
        final_count = collection.count()
        print(f"✓ ChromaDB collection completed. Total items: {final_count}")

    except Exception as e:
        print(f"Error during processing: {e}")
        return

    # write chunk file for training
    print(f"Writing {len(all_chunks)} chunks to {OUT_CHUNKS_FILE}...")
    try:
        with open(OUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(c.replace("\n", " ").strip() + "\n")
        print(f"✓ Chunks written to {OUT_CHUNKS_FILE}")
    except Exception as e:
        print(f"Error writing chunks: {e}")
        return

    # build simple word-level vocab from all_chunks (lowercased)
    print("Building vocab...")
    try:
        words = set()
        for c in all_chunks:
            for w in c.lower().split():
                words.add(w)

        specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        vocab = specials + sorted(words)

        with open(OUT_VOCAB_FILE, "w", encoding="utf-8") as f:
            for w in vocab:
                f.write(w + "\n")
        
        print(f"✓ Vocab written to {OUT_VOCAB_FILE}. Vocab size: {len(vocab)}")
    except Exception as e:
        print(f"Error building vocab: {e}")
        return

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Total chunks processed: {len(all_chunks)}")
    print(f"ChromaDB items: {collection.count()}")
    print(f"Vocab size: {len(vocab)}")
    
    if TEST_MODE:
        print(f"\nTEST MODE was enabled. To process full dataset:")
        print(f"Set TEST_MODE = False at the top of the script")

if __name__ == "__main__":
    main()