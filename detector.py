import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from dataset import SpamMessageDataset
from utils.metrics import compute_metrics, confusion_matrix
from utils.plotting import plot_heatmap
from utils.seed import random_seed

import matplotlib.pyplot as plt

from tqdm import tqdm

def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

class SpamMessageDetector:
    def __init__(self, model_path, max_length=512, seed=0):
        random_seed(seed)
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.model = self.model.to(self.device)
        self.max_length = max_length
    
    def train(self, train_data_path, val_data_path=None, num_epochs=5, batch_size=32, learning_rate=2e-5):
        random_seed(self.seed)

        if(val_data_path is None): # no validation dataset, split the given data
            # Load and preprocess the training data
            data = pd.read_csv(train_data_path)
            text = data['text'].values
            labels = data['label'].values

            # Create the dataset
            dataset = SpamMessageDataset(text, labels, self.tokenizer, max_length=self.max_length)
            # Split the dataset into training and validation sets
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            # Load and preprocess the training data
            train_data = pd.read_csv(train_data_path)
            train_text = train_data['text'].values
            train_labels = train_data['label'].values
            train_dataset = SpamMessageDataset(train_text, train_labels, self.tokenizer, max_length=self.max_length)
            val_data = pd.read_csv(train_data_path)
            val_text = val_data['text'].values
            val_labels = val_data['label'].values
            val_dataset = SpamMessageDataset(val_text, val_labels, self.tokenizer, max_length=self.max_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

        # Fine-tuning loop
        train_losses = list()
        val_losses = list()
        val_accuracies = list()
        val_precisions = list()
        val_recalls = list()
        val_f1_scores = list()

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                
                # Update the progress bar
                progress_bar.set_postfix({'Training Loss': train_loss / (batch_size * (progress_bar.n + 1))})
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Evaluation on the validation set
            self.model.eval()
            val_loss = 0.0
            total_val_loss = 0.0
            val_accuracy = 0.0
            val_precision = 0.0
            val_recall = 0.0

            with torch.no_grad():
                y_true = []
                y_pred = []

                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                    loss = outputs.loss
                    logits = outputs.logits
                    total_val_loss += loss.item()

                    predictions = torch.argmax(logits, dim=1)

                    y_true.extend(labels.tolist())
                    y_pred.extend(predictions.tolist())

                val_loss = total_val_loss / len(val_loader)
                val_losses.append(val_loss)
                val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(y_true, y_pred, 1, 0)
                val_precisions.append(val_precision)
                val_recalls.append(val_recall)
                val_f1_scores.append(val_f1)
                val_accuracies.append(val_accuracy)

            # Print the metrics and confusion matrix for each epoch
            print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - Val Precision: {val_precision:.4f} - Val Recall: {val_recall:.4f}')
        
        # Plots data
        save_list_to_file(train_losses, "plots/train_losses.txt")
        save_list_to_file(val_losses, "plots/val_losses.txt")
        save_list_to_file(val_accuracies, "plots/val_accuracies.txt")
        save_list_to_file(val_precisions, "plots/val_precisions.txt")
        save_list_to_file(val_recalls, "plots/val_recalls.txt")
        save_list_to_file(val_f1_scores, "plots/val_f1_scores.txt")

        # Plots
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('plots/train_validation_loss.jpg')

        plt.figure(figsize=(10, 6))
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig('plots/validation_accuracy.jpg')

        plt.figure(figsize=(10, 6))
        plt.plot(val_precisions, label='Validation Precision')
        plt.plot(val_recalls, label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Precision / Recall')
        plt.title('Precision / Recall')
        plt.legend()
        plt.savefig('plots/validation_precision_recall.jpg')


    def evaluate(self, dataset_path):
        random_seed(self.seed)

        # Load and preprocess the dataset
        dataset = pd.read_csv(dataset_path)
        texts = dataset["text"].tolist()
        labels = dataset["label"].tolist()
        
        def preprocess(text):
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )
            return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)

        inputs = [preprocess(text) for text in texts]

        # Make predictions on the dataset
        predictions = []
        with torch.no_grad():
            for input_ids, attention_mask in inputs:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_label = torch.argmax(logits, dim=1).item()
                if predicted_label == 0:
                    predictions.append("ham")
                else:
                    predictions.append("spam")
                    
        # compute evaluation metrics
        accuracy, precision, recall, f1 = compute_metrics(labels, predictions)

        # Create confusion matrix
        cm = confusion_matrix(labels, predictions)
        labels_sorted = sorted(set(labels))

        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Plot the confusion matrix
        plot_heatmap(cm, saveToFile="plots/confusion_matrix.png", annot=True, fmt="d", cmap="Blues", xticklabels=labels_sorted, yticklabels=labels_sorted)
    
    def detect(self, text):
        random_seed(self.seed)
        is_str = True
        if isinstance(text, str):
            encoded_input = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        elif isinstance(text, list):
            is_str = False
            encoded_input = self.tokenizer.batch_encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            raise Exception("text type is unsupported, needs to be str or list(str)")

        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).tolist()
        
        if is_str: 
            return predicted_labels[0]
        else:
            return predicted_labels
    
    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load_model(self, model_path):
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = self.model.to(self.device)