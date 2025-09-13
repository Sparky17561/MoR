# app.py
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import os

from mor_model import WordTokenizer, MoRLM

# CONFIG
CHROMA_DB_DIR = "chromadb_store"
COLLECTION_NAME = "bitext_chunks"  # Fixed to match ingest.py
EMBED_MODEL = "all-MiniLM-L6-v2"
VOCAB_FILE = "data/vocab.txt"
CHECKPOINT = "checkpoints/mor_checkpoint.pt"
TOP_K = 4
MAX_GEN_TOKENS = 80
TEMPERATURE = 0.8
TOP_K_SAMPLING = 0   # 0 means full sampling (with temperature). set >0 for top-k.

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    """Load all models and resources (cached)"""
    try:
        # Check if required files exist
        if not os.path.exists(VOCAB_FILE):
            st.error(f"Vocab file not found: {VOCAB_FILE}. Please run ingest.py first.")
            st.stop()
        
        if not os.path.exists(CHECKPOINT):
            st.error(f"Model checkpoint not found: {CHECKPOINT}. Please run train_mor.py first.")
            st.stop()
        
        if not os.path.exists(CHROMA_DB_DIR):
            st.error(f"ChromaDB directory not found: {CHROMA_DB_DIR}. Please run ingest.py first.")
            st.stop()
        
        # Load ChromaDB
        with st.spinner("Loading ChromaDB..."):
            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            try:
                collection = client.get_collection(COLLECTION_NAME)
                st.success(f"✓ Loaded ChromaDB collection: {COLLECTION_NAME} ({collection.count()} documents)")
            except Exception as e:
                st.error(f"Failed to load collection '{COLLECTION_NAME}': {e}")
                st.stop()
        
        # Load embedder
        with st.spinner("Loading embedder..."):
            embedder = SentenceTransformer(EMBED_MODEL)
            st.success(f"✓ Loaded embedder: {EMBED_MODEL}")
        
        # Load tokenizer
        with st.spinner("Loading tokenizer..."):
            tokenizer = WordTokenizer(VOCAB_FILE)
            st.success(f"✓ Loaded tokenizer (vocab size: {len(tokenizer.itos)})")
        
        # Load MoR model
        with st.spinner("Loading MoR model..."):
            model = MoRLM.load(CHECKPOINT, device=DEVICE)
            model.to(DEVICE)
            model.eval()
            st.success(f"✓ Loaded MoR model on {DEVICE}")
        
        return client, collection, embedder, tokenizer, model
    
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

def format_retrieved_docs(docs, metas, dists):
    """Format retrieved documents for display"""
    md = "**Retrieved Context:**\n\n"
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        source = meta.get('source', 'unknown')
        category = meta.get('category', '')
        intent = meta.get('intent', '')
        
        # Add metadata info
        meta_info = f"Category: {category}" if category else ""
        if intent and meta_info:
            meta_info += f", Intent: {intent}"
        elif intent:
            meta_info = f"Intent: {intent}"
        
        md += f"**[{i+1}]** *{source}*"
        if meta_info:
            md += f" ({meta_info})"
        md += f" (similarity: {1-dist:.3f})\n"
        md += f"{doc[:400]}{'...' if len(doc) > 400 else ''}\n\n"
    
    return md

def main():
    st.set_page_config(
        page_title="MoR RAG Chat",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 MoR + ChromaDB RAG Chat")
    st.markdown("*Powered by Mixture-of-Recursions Language Model with Retrieval-Augmented Generation*")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Generation Settings")
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, TEMPERATURE, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 10, 200, MAX_GEN_TOKENS, 10)
    top_k_sampling = st.sidebar.slider("Top-K Sampling", 0, 50, TOP_K_SAMPLING, 1)
    
    # RAG settings
    st.sidebar.subheader("RAG Settings")
    retrieval_k = st.sidebar.slider("Retrieved Documents", 1, 10, TOP_K, 1)
    
    # Show system info
    st.sidebar.subheader("System Info")
    st.sidebar.info(f"Device: {DEVICE}")
    
    # Load models
    client, collection, embedder, tokenizer, model = load_models()
    
    st.sidebar.success("✅ All models loaded successfully!")
    
    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.greeted = False

    if not st.session_state.greeted:
        welcome_msg = """👋 Hi! I'm a customer support assistant trained on support conversations. 

I use **Retrieval-Augmented Generation (RAG)** to find relevant information from my knowledge base and then generate responses using a **Mixture-of-Recursions (MoR)** language model.

Ask me anything about customer support, and I'll do my best to help based on what I've learned!"""
        
        st.session_state.history.append(("assistant", welcome_msg))
        st.session_state.greeted = True

    # Display chat history
    for sender, msg in st.session_state.history:
        if sender == "user":
            with st.chat_message("user"):
                st.markdown(msg)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg)

    # Chat input
    if prompt := st.chat_input("Ask me anything about customer support..."):
        # Add user message to history
        st.session_state.history.append(("user", prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Show thinking process
            with st.spinner("Searching knowledge base..."):
                # Retrieve top-k passages
                q_emb = embedder.encode([prompt], convert_to_numpy=True)[0].tolist()
                results = collection.query(
                    query_embeddings=[q_emb], 
                    n_results=retrieval_k, 
                    include=["documents", "metadatas", "distances"]
                )
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results.get("distances", [[]])[0]

            # Show retrieved documents in expandable section
            with st.expander("📋 Retrieved Context (click to expand)", expanded=False):
                retrieval_md = format_retrieved_docs(docs, metas, dists)
                st.markdown(retrieval_md)

            # Generate response
            with st.spinner("Generating response..."):
                # Build context prompt
                context_parts = []
                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    context_parts.append(f"[{i+1}] {doc}")
                context_text = " ".join(context_parts)
                
                # Create instruction-following prompt
                instruction = "answer the question using only the provided context. if the answer is not found in the context, say 'i don't know based on the provided information'."
                full_prompt = f"{instruction} CONTEXT: {context_text} QUESTION: {prompt} ANSWER:"

                # Generate using MoR model
                try:
                    response = model.generate(
                        tokenizer, 
                        full_prompt, 
                        max_new_tokens=max_tokens, 
                        device=DEVICE, 
                        temperature=temperature, 
                        top_k=top_k_sampling if top_k_sampling > 0 else 0
                    )
                    
                    # Clean up response
                    response = response.strip()
                    if not response:
                        response = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state.history.append(("assistant", response))
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while generating a response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.history.append(("assistant", error_msg))

    # Footer with information
    st.markdown("---")
    with st.expander("ℹ️ About this system", expanded=False):
        st.markdown("""
        **This system combines:**
        - **Retrieval-Augmented Generation (RAG)**: Searches a knowledge base of customer support conversations
        - **Mixture-of-Recursions (MoR)**: A novel transformer architecture that adaptively processes information
        - **ChromaDB**: Vector database for efficient similarity search
        - **Sentence Transformers**: For encoding queries and documents into embeddings
        
        **How it works:**
        1. Your question is encoded into a vector using Sentence Transformers
        2. ChromaDB finds the most similar support conversations from the knowledge base
        3. The retrieved context is fed to the MoR language model
        4. MoR generates a response based on the retrieved information
        
        **Model Details:**
        - **Dataset**: Bitext Customer Support conversations
        - **Embedding Model**: all-MiniLM-L6-v2
        - **Language Model**: Custom MoR architecture with adaptive computation
        - **Vector DB**: ChromaDB for persistent storage and fast retrieval
        """)

    # Clear chat history button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.history = []
        st.session_state.greeted = False
        st.rerun()

if __name__ == "__main__":
    main()