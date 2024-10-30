import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loader import load_training_data, load_validation_data
from config import Config
from utils import arguments


class KasrePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(KasrePredictor, self).__init__()
        
        # Character embeddings
        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # +1 for boundary information
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 2)  # *2 for bidirectional
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, char_ids, boundaries):
        # Get character embeddings
        char_embeds = self.char_embeddings(char_ids)  # [batch, seq_len, embed_dim]
        
        # Concatenate boundary information
        boundaries = boundaries.unsqueeze(-1)  # [batch, seq_len, 1]
        lstm_input = torch.cat([char_embeds, boundaries], dim=-1)
        
        # Apply dropout to the input
        lstm_input = self.dropout(lstm_input)
        
        # LSTM layer
        lstm_out, _ = self.lstm(lstm_input)
        
        # Apply dropout to LSTM output
        lstm_out = self.dropout(lstm_out)
        
        # Project to output space
        logits = self.fc(lstm_out)
        
        return logits

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    # Load pre-trained model if it exists
    if os.path.exists('best_model.pt'):
        print("Loading pre-trained model 'best_model.pt' for fine-tuning...")
        model.load_state_dict(torch.load('best_model.pt', weights_only=True))



    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            char_ids = batch['sentence'].to(device)
            boundaries = batch['boundary'].to(device)
            targets = batch['diacritics'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(char_ids, boundaries)
            loss = criterion(outputs.view(-1, 2), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                char_ids = batch['sentence'].to(device)
                boundaries = batch['boundary'].to(device)
                targets = batch['diacritics'].to(device)
                
                outputs = model(char_ids, boundaries)
                loss = criterion(outputs.view(-1, 2), targets.view(-1))
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')

def predict(model, char_ids, boundaries, device):
    model.eval()
    with torch.no_grad():
        char_ids = char_ids.to(device)
        boundaries = boundaries.to(device)
        outputs = model(char_ids, boundaries)
        predictions = torch.argmax(outputs, dim=-1)
    return predictions

# Example usage:
if __name__ == "__main__":
    args = arguments().parse_args()
    config_path = args.config
    config = Config(config_path)

    # Hyperparameters
    VOCAB_SIZE = 63  # Adjust based on your character vocabulary
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = KasrePredictor(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Create sample data loaders
    # Replace these with your actual data
    train_loader = load_training_data(config)
    valid_loader = load_validation_data(config)

    train_model(model, train_loader, valid_loader, NUM_EPOCHS, LEARNING_RATE, device)