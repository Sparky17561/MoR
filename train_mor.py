"""
Train the MoR LM on the chunked text saved by ingest.py (data/chunks.txt).
Saves model checkpoint and keeps vocab (data/vocab.txt must exist).
OPTIMIZED FOR SUBSET TRAINING - FAST 2-3 HOUR TRAINING!
ENHANCED: Resume from checkpoint + GPU optimization for RTX 3050 Ti
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json
import time

from mor_model import WordTokenizer, MoRLM

# CONFIG - OPTIMIZED FOR RTX 3050 Ti (8GB VRAM)
CHUNKS_FILE = "data/chunks.txt"
VOCAB_FILE = "data/vocab.txt"
CHECKPOINT = "checkpoints/mor_checkpoint.pt"
TRAINING_STATE_FILE = "checkpoints/training_state.json"
OPTIMIZER_STATE_FILE = "checkpoints/optimizer_state.pt"
BATCH_SIZE = 24          # Optimized for 3050 Ti VRAM
BLOCK_SIZE = 64          # Good context
EPOCHS = 12              # Good for subset
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_CLIP = 1.0
VAL_SPLIT = 0.1          # 10% for validation
SAVE_BEST = True         # save best model based on validation loss
PATIENCE = 5             # early stopping patience
TEST_MODE = False            # set to True for quick test run
TEST_TOKENS = 1000     # Only use 50K tokens in test mode
# FAST TRAINING MODE - USE SUBSET
USE_SUBSET = True
SUBSET_TOKENS = 500000   # Use 500K tokens instead of full 3.3M (much faster!)

# Resume settings
RESUME_FROM_CHECKPOINT = True  # Set to True to resume from last checkpoint

os.makedirs("checkpoints", exist_ok=True)

def save_training_state(epoch, best_val_loss, patience_counter, optimizer_state_dict):
    """Save training state for resuming"""
    # Save optimizer state separately as a PyTorch file
    torch.save(optimizer_state_dict, OPTIMIZER_STATE_FILE)
    
    # Save JSON-serializable training state
    state = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'timestamp': time.time()
    }
    
    with open(TRAINING_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"Training state saved: epoch {epoch}")

def load_training_state():
    """Load training state for resuming"""
    if not os.path.exists(TRAINING_STATE_FILE):
        return None
    
    try:
        with open(TRAINING_STATE_FILE, 'r') as f:
            state = json.load(f)
        
        print(f"Found training state from epoch {state['epoch']}")
        return state
    except Exception as e:
        print(f"Failed to load training state: {e}")
        return None

def check_gpu_setup():
    """Check and optimize GPU setup for RTX 3050 Ti"""
    print("=" * 50)
    print("GPU SETUP CHECK")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! Running on CPU (will be much slower)")
        return "cpu"
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✓ GPU detected: {gpu_name}")
    print(f"✓ GPU memory: {gpu_memory:.1f} GB")
    
    # Optimize for RTX 3050 Ti (4GB VRAM)
    if "3050" in gpu_name or gpu_memory < 6:
        print("🎯 RTX 3050 Ti detected - optimizing for 4GB VRAM:")
        print(f"  - Batch size: {BATCH_SIZE} (optimized for 4GB VRAM)")
        print("  - Mixed precision: Enabled")
        print("  - Memory optimization: Enabled")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    return "cuda"

def test_setup():
    """Test basic setup before training"""
    print("=" * 50)
    print("TESTING SETUP...")
    print("=" * 50)
    
    try:
        # Check files exist
        if not os.path.exists(VOCAB_FILE):
            print(f"✗ Vocab file not found: {VOCAB_FILE}")
            return False
        
        if not os.path.exists(CHUNKS_FILE):
            print(f"✗ Chunks file not found: {CHUNKS_FILE}")
            return False
        
        # Test tokenizer
        tokenizer = WordTokenizer(VOCAB_FILE)
        print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer.itos)}")
        
        # Test model creation
        model = MoRLM(vocab_size=len(tokenizer.itos), dim=256, n_heads=4, mlp_hidden=512, max_depth=4)
        print(f"✓ Model created successfully")
        
        # Test generation
        test_output = model.generate(tokenizer, "test prompt", max_new_tokens=5, device="cpu")
        print(f"✓ Model generation works: '{test_output}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Setup test failed: {e}")
        return False

# create sliding windows dataset
class WordSeqDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

def validate_model(model, val_loader, device, scaler=None):
    """Run validation and return average loss"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            if scaler and device == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(xb)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(1, num_batches)

def test_generation(model, tokenizer, device):
    """Test model generation with sample prompts"""
    test_prompts = ["How can I help", "What is your", "I need to"]
    
    print("\nGeneration test:")
    model.eval()
    for prompt in test_prompts:
        try:
            output = model.generate(tokenizer, prompt, max_new_tokens=15, device=device, temperature=0.7)
            print(f"  '{prompt}' → '{output}'")
        except Exception as e:
            print(f"  '{prompt}' → Error: {e}")

def main():
    print("=" * 60)
    print("MoR LANGUAGE MODEL TRAINING - ENHANCED VERSION")
    print("=" * 60)
    
    # Check and optimize GPU setup
    device = check_gpu_setup()
    print(f"Using device: {device}")
    
    if RESUME_FROM_CHECKPOINT:
        print(f"Resume mode: {'ON' if RESUME_FROM_CHECKPOINT else 'OFF'}")
    
    print(f"Subset Training: {USE_SUBSET}")
    print(f"Target Tokens: {SUBSET_TOKENS:,} (instead of full 3.3M)")
    print(f"Estimated Time: 2-3 hours (instead of 15+ hours)")
    
    # Run setup test first
    if not test_setup():
        print("Setup test failed. Please fix issues before proceeding.")
        return
    
    # load tokenizer
    print(f"\nLoading tokenizer from {VOCAB_FILE}...")
    tokenizer = WordTokenizer(VOCAB_FILE)
    vocab_size = len(tokenizer.itos)
    print(f"Vocabulary size: {vocab_size}")

    # load chunks and tokenize into long sequence of word tokens
    print(f"Loading chunks from {CHUNKS_FILE}...")
    all_token_ids = []
    chunk_count = 0
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            toks = tokenizer.encode(text) + ([tokenizer.eos_id] if tokenizer.eos_id is not None else [])
            all_token_ids.extend(toks)
            chunk_count += 1

    print(f"Loaded {chunk_count} chunks, {len(all_token_ids):,} total tokens")
    
    # Apply subset for fast training
    if TEST_MODE:
        original_length = len(all_token_ids)
        all_token_ids = all_token_ids[:TEST_TOKENS]
        print(f"🧪 TEST MODE: Using {len(all_token_ids):,} tokens out of {original_length:,} total ({len(all_token_ids)/original_length:.1%})")
        print(f"Estimated training time: ~5-10 minutes for testing")
    elif USE_SUBSET:
        original_length = len(all_token_ids)
        all_token_ids = all_token_ids[:SUBSET_TOKENS]
        print(f"SUBSET MODE: Using {len(all_token_ids):,} tokens out of {original_length:,} total ({len(all_token_ids)/original_length:.1%})")
        print(f"Estimated training time: ~2-3 hours instead of ~15 hours")

    if len(all_token_ids) < BLOCK_SIZE + 1:
        raise ValueError(f"Not enough tokens ({len(all_token_ids)}) to form training examples with BLOCK_SIZE={BLOCK_SIZE}. Check ingest output.")

    # Create train/val split
    split_idx = int(len(all_token_ids) * (1 - VAL_SPLIT))
    train_tokens = all_token_ids[:split_idx]
    val_tokens = all_token_ids[split_idx:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

    # Create datasets
    train_dataset = WordSeqDataset(train_tokens, BLOCK_SIZE)
    val_dataset = WordSeqDataset(val_tokens, BLOCK_SIZE) if len(val_tokens) > BLOCK_SIZE else None
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device=="cuda")) if val_dataset else None
    
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples: {len(val_dataset) if val_dataset else 0:,}")

    # instantiate model
    model = MoRLM(vocab_size=vocab_size, dim=256, n_heads=4, mlp_hidden=512, max_depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Mixed precision for RTX 3050 Ti
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Try to resume from checkpoint
    if RESUME_FROM_CHECKPOINT:
        training_state = load_training_state()
        
        if training_state and os.path.exists(CHECKPOINT):
            try:
                print("🔄 Resuming from checkpoint...")
                
                # Load model
                checkpoint = torch.load(CHECKPOINT, map_location=device)
                if hasattr(checkpoint, 'state_dict'):
                    model.load_state_dict(checkpoint.state_dict())
                else:
                    model.load_state_dict(checkpoint)
                
                # Restore training state
                start_epoch = training_state['epoch'] + 1
                best_val_loss = training_state['best_val_loss']
                patience_counter = training_state['patience_counter']
                
                # Load optimizer state if available
                if os.path.exists(OPTIMIZER_STATE_FILE):
                    try:
                        optimizer_state_dict = torch.load(OPTIMIZER_STATE_FILE, map_location=device)
                        opt.load_state_dict(optimizer_state_dict)
                        print("✓ Optimizer state restored")
                    except Exception as e:
                        print(f"⚠️  Could not restore optimizer state: {e}")
                        print("Using fresh optimizer...")
                
                print(f"✓ Resumed from epoch {start_epoch}")
                print(f"✓ Best validation loss so far: {best_val_loss:.4f}")
                
            except Exception as e:
                print(f"⚠️  Failed to resume from checkpoint: {e}")
                print("Starting fresh training...")
                start_epoch = 0
                best_val_loss = float('inf')
                patience_counter = 0
        else:
            print("No valid checkpoint found, starting fresh training...")

    # Test initial generation
    test_generation(model, tokenizer, device)

    print(f"\nStarting training from epoch {start_epoch + 1}...")
    print("=" * 60)
    
    try:
        for epoch in range(start_epoch, EPOCHS):
            epoch_start_time = time.time()
            
            # Training
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            total_loss = 0.0
            
            for batch_idx, (xb, yb) in enumerate(loop):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                
                if scaler and device == "cuda":
                    # Mixed precision training - FIXED deprecation warning
                    with torch.amp.autocast('cuda'):
                        logits = model(xb)   # (B, T, V)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                    
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()
                else:
                    # Regular training
                    logits = model(xb)   # (B, T, V)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
                # Clear cache periodically to prevent OOM
                if device == "cuda" and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            avg_train_loss = total_loss / max(1, len(train_loader))
            epoch_time = time.time() - epoch_start_time
            
            # Validation
            if val_loader:
                avg_val_loss = validate_model(model, val_loader, device, scaler)
                
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, time={epoch_time:.1f}s")
                
                # Save best model
                if SAVE_BEST and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_checkpoint = CHECKPOINT.replace('.pt', '_best.pt')
                    model.save(best_checkpoint)
                    print(f"  → New best model saved: {best_checkpoint}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= PATIENCE:
                    print(f"Early stopping after {patience_counter} epochs without improvement")
                    break
            else:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, time={epoch_time:.1f}s")
            
            # Save regular checkpoint and training state - FIXED JSON serialization
            model.save(CHECKPOINT)
            save_training_state(epoch, best_val_loss, patience_counter, opt.state_dict())
            
            # Test generation every few epochs
            if (epoch + 1) % max(1, EPOCHS // 3) == 0:
                test_generation(model, tokenizer, device)
            
            # Clear cache at end of epoch
            if device == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 60)
        print("Saving current progress...")
        
        # Save final checkpoint and state
        model.save(CHECKPOINT)
        save_training_state(epoch, best_val_loss, patience_counter, opt.state_dict())
        
        print(f"✓ Progress saved! Resume with same command.")
        print(f"✓ Last completed epoch: {epoch+1}")
        return

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Final checkpoint saved to: {CHECKPOINT}")
    
    if SAVE_BEST and val_loader:
        print(f"Best model (val_loss={best_val_loss:.4f}) saved to: {best_checkpoint}")
    
    # Clean up training state files
    if os.path.exists(TRAINING_STATE_FILE):
        os.remove(TRAINING_STATE_FILE)
        print("✓ Training state file cleaned up")
    
    if os.path.exists(OPTIMIZER_STATE_FILE):
        os.remove(OPTIMIZER_STATE_FILE)
        print("✓ Optimizer state file cleaned up")
    
    # Final generation test
    print("\nFinal generation test:")
    test_generation(model, tokenizer, device)
    
    print("\n🎉 SUBSET TRAINING COMPLETE!")
    print(f"📊 Trained on {len(all_token_ids):,} tokens")
    print(f"🚀 Ready to test with: streamlit run app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise