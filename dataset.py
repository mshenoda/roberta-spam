import torch
from torch.utils.data import Dataset

class SpamMessageDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_length):
        self.text = text
        labels = [1 if label == 'spam' else 0 for label in labels]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = str(self.text[idx])
        label = self.labels[idx].clone().detach()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }