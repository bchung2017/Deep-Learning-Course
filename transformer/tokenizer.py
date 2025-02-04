import re
from collections import defaultdict
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class BytePairEncoding:
    def __init__(self, data, merges):
        self.merges = merges
        self.merges_rules = {}
        self.vocab = {}
        self.byte_pair_encoding(data)
        self.build_vocab(data)

    def get_pair_frequency(self, vocab):
        byte_pairs = defaultdict(int)
        for word, freq in vocab.items():
            characters = word.split()
            for i in range(len(characters) - 1):
                byte_pairs[(characters[i], characters[i + 1])] += freq
        return byte_pairs

    def merge_vocab(self, byte_pair, vocab_in):
        vocab_out = {}
        bigram = ' '.join(byte_pair)
        replacement = ''.join(byte_pair)

        for word in vocab_in:
            word_parts = word.split()
            merged_word = []
            i = 0
            while i < len(word_parts):
                if i < len(word_parts) - 1 and word_parts[i] == byte_pair[0] and word_parts[i + 1] == byte_pair[1]:
                    merged_word.append(replacement)
                    i += 2
                else:
                    merged_word.append(word_parts[i])
                    i += 1
            vocab_out[' '.join(merged_word)] = vocab_in[word]

        return vocab_out

    def get_vocab(self, data):
        vocab = defaultdict(int)
        for line in data:
            for word in line.split():
                vocab[' '.join(list(word)) + ' </w>'] += 1
        return vocab

    def byte_pair_encoding(self, data):
        vocab = self.get_vocab(data)
        for _ in range(self.merges):
            pairs = self.get_pair_frequency(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges_rules[best_pair] = ''.join(best_pair)

    def build_vocab(self, data):
        word_set = set()
        for sentence in data:
            tokens = sentence.split()
            word_set.update(tokens)
        
        self.vocab = {word: idx for idx, word in enumerate(word_set)}
        self.vocab["<EOS>"] = 0 
        
        print("Built vocabulary:", list(self.vocab.items())[:10]) 

    def tokenize(self, text):
        tokens = [' '.join(list(word)) + ' </w>' for word in text.split()]
        for _ in range(self.merges):
            pairs = self.get_pair_frequency({t: 1 for t in tokens})
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            tokens = [list(self.merge_vocab(best_pair, {t: 1}).keys())[0] for t in tokens]
        return tokens

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens if token in self.vocab]

class Tokenizer:
    def __init__(self, data, merges=230):
        self.data = data
        self.vocab = {}
        self.build_vocab(data)

    def build_vocab(self, data):
        word_set = set()
        for sentence in data:
            tokens = sentence.split()
            word_set.update(tokens)
        
        self.vocab = {word: idx for idx, word in enumerate(word_set)}
        
        print("Built vocabulary sample:", list(self.vocab.items())[:10])

    def tokenize(self, text):
        tokens = text.split()
        token_ids = [self.vocab.get(token, None) for token in tokens]
        
        token_ids = [id for id in token_ids if id is not None]

        return token_ids
    
    def tokenize_dataset(self, dataset, column_name):
        tokenized_data = dataset.map(lambda x: {f"{column_name}_tokens": self.tokenize(x[column_name])}, batched=False)

        print("Tokenized data sample:")
        for i in range(min(5, len(tokenized_data))):
            original_text = tokenized_data[i][column_name]
            tokenized_output = tokenized_data[i][f"{column_name}_tokens"]
            print(f"Original text {i}: {original_text}")
            print(f"Tokenized output {i}: {tokenized_output}")

        return tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx][f"{self.column_name}_tokens"]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        target_ids = [item[1] for item in batch]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
        return input_ids_padded, target_ids_padded

    def create_tokenized_dataloader(self, dataset, column_name="text", batch_size=32, shuffle=True):
        tokenized_data = self.tokenize_dataset(dataset, column_name=column_name)
        tokenized_dataset = TokenizedDataset(tokenized_data, column_name=column_name)
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


    
    def decode(self, token_ids):
        id_to_token = {id: token for token, id in self.vocab.items()}
        decoded_tokens = [id_to_token[token_id] for token_id in token_ids if token_id in id_to_token]
        
        return " ".join(map(str, decoded_tokens))



class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data, column_name="text"):
        self.data = tokenized_data
        self.column_name = column_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][f"{self.column_name}_tokens"]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return torch.tensor(input_ids), torch.tensor(target_ids)
