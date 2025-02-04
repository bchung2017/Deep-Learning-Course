from datasets import load_dataset
import torch
import torch.nn as nn
from tokenizer import Tokenizer
from torch.utils.data import DataLoader, Dataset
from model import NextWordPredictionModel
import torch.nn.functional as F



def main():
    raw_dataset = load_dataset("ag_news", split="train[:1000]")

    raw_text_data = [entry["text"] for entry in raw_dataset]
    
    tokenizer = Tokenizer(raw_text_data)
    tokenizer.build_vocab(raw_text_data)
    dataloader = tokenizer.create_tokenized_dataloader(raw_dataset, column_name="text", batch_size=32, shuffle=True)
    for input_ids, target_ids in dataloader:
        print("Input IDs:", input_ids)
        print("Target IDs:", target_ids)
        break

    vocab_size = len(tokenizer.vocab) + 1
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    hidden_dim = 2048
    model = NextWordPredictionModel(vocab_size, embed_dim, num_layers, num_heads, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, target_ids in dataloader:
            optimizer.zero_grad()
            batch_size = input_ids.size(0)
            seq_length = input_ids.size(1)
            src_mask = model.create_mask(seq_length, batch_size=batch_size, num_heads=model.num_heads)
            logits = model(input_ids, src_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved")

    model.eval()
    input_text = "This is a"
    input_tokens = tokenizer.tokenize(input_text)
    print("Tokenized input:", input_tokens)
    input_ids = torch.tensor([input_tokens], dtype=torch.long)

    num_words_to_predict = 20
    predicted_text = input_text


    for _ in range(num_words_to_predict):
        src_mask = model.create_mask(input_ids.size(1), batch_size=1, num_heads=model.num_heads)

        with torch.no_grad():
            logits = model(input_ids, src_mask)
            next_token_logits = logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.argmax(next_token_probs, dim=-1).item()

        next_word = tokenizer.decode([next_token_id])
        
        predicted_text += " " + next_word

        input_tokens.append(next_token_id)
        input_ids = torch.tensor([input_tokens], dtype=torch.long)

    print(f"Input: '{input_text}'")
    print(f"Predicted sentence: '{predicted_text}'")

if __name__ == "__main__":
    main()