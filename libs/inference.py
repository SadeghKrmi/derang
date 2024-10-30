import torch
import numpy as np
from model import KasrePredictor
from encoder import TextEncoder

def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers, device):
    model = KasrePredictor(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def predict(model, char_ids, boundaries, device):
    model.eval()
    with torch.no_grad():
        char_ids = char_ids.to(device)
        boundaries = boundaries.to(device)
        outputs = model(char_ids, boundaries)
        predictions = torch.argmax(outputs, dim=-1)
    return predictions

# Example Usage
if __name__ == "__main__":
    # Configurations
    VOCAB_SIZE = 63
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model = load_model('best_model.pt', VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DEVICE)
    
    # Example input (tokenized sentence and boundaries)
    text_encoder = TextEncoder()
    text = 'کتاب اصلی و جزوه درسی را در راه خانه خواندم'
    text = 'درس زندگی چیست؟'
    text = 'خواندن سرود ملی خیلی زیباست'    # 1 mistake(1/2)
    text = 'روز آفتابی خوبی برای شما دوست عزیز خواستارم'    # 1 mistake(1/3)
    text = 'معرفی ارزون‌ترین و بهترین کراپا توسط زن موتورسوار!'  # 1 mistake(1/3)
    text = 'قلاب زن موتورسوار در حال حاضر یکی از بهترین قلاب‌هاست!'  # 1 mistake(1/3)
    text = 'چند خبر کوتاه از امروز' # 1 mistake(1/1)

    sample_sentence = text_encoder.input_to_sequence(text)
    sample_sentence_tensor = torch.tensor(sample_sentence, dtype=torch.long).unsqueeze(0)

    sample_boundary = text_encoder.input_to_word_boundary(text)
    sample_boundary_tensor = torch.tensor(sample_boundary, dtype=torch.long).unsqueeze(0)
    
    # Run inference
    diacritic_predictions = predict(model, sample_sentence_tensor, sample_boundary_tensor, DEVICE)
    print("Predicted diacritic positions:", diacritic_predictions)
    print(diacritic_predictions.tolist())
    print(len(sample_sentence))
    predicted_text = text_encoder.combine_text_and_diacritics(sample_sentence, diacritic_predictions[0].tolist())
    print(predicted_text)