import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

# Page config
st.set_page_config(
    page_title="Goldfish AI",
    page_icon="🐠",
    layout="wide"
)

# Enhanced CSS for modern chat UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px auto;
        max-width: 800px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        min-height: 500px;
    }
    
    /* Title styling */
    .app-title {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .app-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Chat messages */
    .message {
        margin: 15px 0;
        padding: 15px 20px;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        animation: fadeIn 0.3s ease-in;
    }
    
    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .bot-msg {
        background: linear-gradient(135deg, #ffeaa7, #fab1a0);
        color: #2d3436;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
    
    /* Input styling */
    .input-container {
        background: #f8f9fa;
        border-radius: 25px;
        padding: 10px;
        margin: 20px 0;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .input-container:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling */
    .send-btn {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px 25px !important;
        transition: all 0.3s ease !important;
    }
    
    .send-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Quick buttons */
    .quick-btn {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 2px solid #667eea !important;
        color: #667eea !important;
        border-radius: 15px !important;
        padding: 8px 16px !important;
        margin: 5px !important;
        transition: all 0.3s ease !important;
    }
    
    .quick-btn:hover {
        background: #667eea !important;
        color: white !important;
        transform: translateY(-2px) !important;
    }
    
    /* Training screen */
    .training-container {
        text-align: center;
        padding: 50px 20px;
        color: white;
    }
    
    .training-title {
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    
    .training-desc {
        font-size: 1.1rem;
        margin-bottom: 30px;
        opacity: 0.9;
    }
    
    /* Welcome message */
    .welcome-msg {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        border-radius: 15px;
        color: white;
        margin: 20px 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Scrollbar */
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    /* Remove Streamlit branding */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ========== MIXTURE OF RECURSIONS (MoR) MODEL ARCHITECTURE ==========
class MoRBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, inner_recursions=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.inner_recursions = inner_recursions

    def forward(self, x):
        for _ in range(max(1, self.inner_recursions)):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
        return x

class MoRLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([MoRBlock(embed_dim) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.fc(x)
        return x

class WordDataset(Dataset):
    def __init__(self, data, block_size=8):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# Utility functions
def load_dataset():
    try:
        with open('dataset.txt', 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            raise FileNotFoundError
        return text
    except FileNotFoundError:
        return """Goldfish are freshwater fish that belong to the carp family. They are native to East Asia and were domesticated over a thousand years ago. Goldfish come in many colors including orange, white, black, and calico patterns. They are social animals that can recognize their owners. Goldfish are omnivorous and eat both plants and small animals. Their diet includes algae, insects, larvae, and fish food. Proper goldfish care requires clean water, good filtration, and adequate space. Goldfish can live over ten years with some reaching twenty years of age. They are popular aquarium fish worldwide due to their hardy nature and cultural significance."""

def create_tokenizer(text):
    words = text.lower().split()
    unique_words = sorted(set(words))
    specials = ["<PAD>", "<UNK>", "<EOS>"]
    vocab = specials + unique_words
    
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    
    UNK_ID = stoi["<UNK>"]
    EOS_ID = stoi["<EOS>"]
    
    def encode(s):
        return [stoi.get(w, UNK_ID) for w in str(s).lower().split()]

    def decode(ids):
        words_out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i >= 0 and i < len(itos):
                word = itos[i]
                if word not in ["<PAD>", "<UNK>"]:
                    words_out.append(word)
        return " ".join(words_out)
    
    return vocab, encode, decode, EOS_ID

def generate_response(model, encode, decode, prompt, EOS_ID, vocab_size, max_tokens=25):
    model.eval()
    with torch.no_grad():
        context_ids = encode(prompt)
        if not context_ids:
            return "I don't understand. Please ask about goldfish!"
        
        input_ids = torch.tensor([context_ids], dtype=torch.long)
        generated = []

        for _ in range(max_tokens):
            logits = model(input_ids)
            next_logits = logits[:, -1, :]
            
            # Apply temperature and sample
            next_logits = next_logits / 0.7
            probs = torch.softmax(next_logits, dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())
            
            if next_token_id == EOS_ID:
                break
                
            generated.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
        
        response = decode(generated)
        if not response or len(response.strip()) < 3:
            return "Goldfish are fascinating aquarium fish that need proper care and clean water to thrive!"
        
        return response.capitalize() + "."

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'training' not in st.session_state:
    st.session_state.training = False

def main():
    # App header
    st.markdown('<div class="app-title">🐠 Goldfish AI Expert</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Your friendly AI companion for all things goldfish!</div>', unsafe_allow_html=True)

    # Main chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Training phase
        if not st.session_state.model_loaded and not st.session_state.training:
            st.markdown("""
            <div class="welcome-msg">
                <h2>🎉 Welcome to Goldfish AI Expert!</h2>
                <p>I'm ready to answer all your goldfish questions, but first I need to learn!</p>
                <p>Click below to train me on goldfish knowledge (takes ~30 seconds)</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Train AI Model", key="train_btn", help="Start training the model"):
                    st.session_state.training = True
                    st.rerun()
        
        # Training in progress
        elif st.session_state.training:
            st.markdown("""
            <div class="training-container">
                <div class="training-title">🧠 Training AI Brain...</div>
                <div class="training-desc">Teaching me everything about goldfish care, behavior, and biology!</div>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training process
            text = load_dataset()
            vocab, encode, decode, EOS_ID = create_tokenizer(text)
            data = torch.tensor(encode(text) + [EOS_ID], dtype=torch.long)
            dataset = WordDataset(data, block_size=6)
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            model = MoRLM(vocab_size=len(vocab), embed_dim=32, num_layers=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            epochs = 5
            for epoch in range(epochs):
                for batch_idx, (xb, yb) in enumerate(loader):
                    logits = model(xb)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.markdown(f"**Learning goldfish facts... {int(progress * 100)}% complete**")
            
            # Save model
            st.session_state.model = model
            st.session_state.encode = encode
            st.session_state.decode = decode
            st.session_state.EOS_ID = EOS_ID
            st.session_state.vocab_size = len(vocab)
            st.session_state.model_loaded = True
            st.session_state.training = False
            
            st.success("🎉 Training complete! I'm ready to chat!")
            time.sleep(1)
            st.rerun()
        
        # Chat interface
        elif st.session_state.model_loaded:
            # Display chat history
            if not st.session_state.chat_history:
                st.markdown("""
                <div class="welcome-msg">
                    <h3>👋 Hello! I'm your Goldfish Expert</h3>
                    <p>Ask me anything about goldfish care, behavior, diet, or biology!</p>
                    <p><em>Try asking: "What do goldfish eat?" or "How long do goldfish live?"</em></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Chat messages
            chat_container = st.container()
            with chat_container:
                for user_msg, bot_msg in st.session_state.chat_history:
                    st.markdown(f'<div class="message user-msg">🙋‍♀️ <strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="message bot-msg">🐠 <strong>AI:</strong> {bot_msg}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Input area
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message", 
                    placeholder="Ask me about goldfish... 🐠",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_clicked = st.button("Send 🚀", key="send", use_container_width=True)
            
            # Process message
            if (send_clicked and user_input.strip()) or (user_input and len(user_input) > 0 and user_input.endswith('\n')):
                if user_input.strip():
                    with st.spinner("🤔 Thinking about goldfish..."):
                        response = generate_response(
                            st.session_state.model,
                            st.session_state.encode,
                            st.session_state.decode,
                            user_input.strip(),
                            st.session_state.EOS_ID,
                            st.session_state.vocab_size
                        )
                        
                        st.session_state.chat_history.append((user_input.strip(), response))
                        st.rerun()
            
            # Quick questions
            st.markdown("### 💡 Quick Questions:")
            col1, col2 = st.columns(2)
            
            questions = [
                "What do goldfish eat?",
                "How long do goldfish live?", 
                "Where are goldfish from?",
                "How to care for goldfish?"
            ]
            
            for i, question in enumerate(questions):
                col = col1 if i % 2 == 0 else col2
                with col:
                    if st.button(question, key=f"q_{i}", use_container_width=True):
                        with st.spinner("🤔 Thinking..."):
                            response = generate_response(
                                st.session_state.model,
                                st.session_state.encode,
                                st.session_state.decode,
                                question,
                                st.session_state.EOS_ID,
                                st.session_state.vocab_size
                            )
                            st.session_state.chat_history.append((question, response))
                            st.rerun()
            
            # Clear button
            if st.session_state.chat_history:
                st.markdown("---")
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

if __name__ == "__main__":
    main()