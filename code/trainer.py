#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from lib.model import PriceGraph
from lib.model import output_layer
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import yaml
from dataset import load_dataframe, sample_by_dates
import json  # Import the json module
from multiprocessing import Lock

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create a lock for controlling print output
print_lock = Lock()

class Trainer:
    def __init__(self, config, args=None):
        self.config = config
        self.args = args  # Store parsed arguments
        self.time_step = config['data']['time_step']
        self.hidden_size = config['model']['hidden_size']
        self.learning_rate = config['model']['learning_rate']
        self.batch_size = config['model']['batch_size']
        self.drop_ratio = config['model']['dropout_ratio']
        # self.validation_ratio = config['model']['validation_split'] #Not directly used.
        self.l2_regularization = config['model']['l2_regularization']
        self.decay_rate = config['model']['decay_rate']
        self.epochs = config['model']['epochs']
        self.save_interval = config['model']['save_interval']
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        with print_lock:
            print(self.device)  # Print the device being used

        # Load dataframes (train, validation, test)
        with print_lock:
            print("Loading dataframes...")
        self.train_df = load_dataframe('train_df')
        self.validation_df = load_dataframe('validation_df')
        self.test_df = load_dataframe('test_df')

        if self.train_df is None or self.validation_df is None or (args and args.test and self.test_df is None):
            raise ValueError("Failed to load one or more dataframes.")

        with print_lock:
            print("Dataframes loaded.")

        # Create datasets by sampling.  This now happens *inside* the trainer,
        # *after* loading the raw data, and *before* training.
        with print_lock:
            print("Sampling data...")
        self.train_data = sample_by_dates(self.train_df, self.time_step)
        self.validation_data = sample_by_dates(self.validation_df, self.time_step)

        # If in test mode, load/sample test data.  Otherwise, set to None.
        if args is not None and args.test:
            self.test_data = sample_by_dates(self.test_df, self.time_step)
            with print_lock:
                print("Test data sampled.")
        else:
            self.test_data = None

        with print_lock:
          print("Data sampled.")

        # Load embeddings and CI.  This happens *after* sampling.
        with print_lock:
            print("Loading embeddings and CI values for training data...")
        self.train_data = self.load_embeddings_and_ci(self.train_data, 'train')
        with print_lock:
            print("Embeddings and CI values loaded for training data.")
        with print_lock:
            print("Loading embeddings and CI values for validation data...")
        self.validation_data = self.load_embeddings_and_ci(self.validation_data, 'validation')
        with print_lock:
            print("Embeddings and CI values loaded for validation data.")

        if args is not None and args.test:
            with print_lock:
              print("Loading embeddings and CI values for test data...")
            self.test_data = self.load_embeddings_and_ci(self.test_data, 'test')
            with print_lock:
              print("Embeddings and CI values loaded for test data.")

        # Determine feature size *after* sampling.
        feature_size = len(self.train_data['close_ys'][0])
        self.feature_size = feature_size
        self.num_features = len(config['data']['features'])
        with print_lock:
            print(f"Number of features: {self.num_features}, Feature size: {feature_size}")
        with print_lock:
            print("Initializing model and optimizer...")
        # Initialize model and optimizer
        self.emtree = PriceGraph(feature_size, self.hidden_size, self.time_step, self.drop_ratio, self.num_features).to(self.device)
        self.output = output_layer(last_hidden_size=self.hidden_size, output_size=3).to(self.device)

        self.emtree_optim = optim.Adam(self.emtree.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization)
        self.output_optim = optim.Adam(self.output.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization)
        self.loss_func = nn.CrossEntropyLoss()
        self.model_name = "price_graph" # Simplified model name
        with print_lock:
            print("Model and optimizer initialized.")

    def load_embeddings_and_ci(self, data, dataset_type):
        """Loads pre-computed embeddings and CI values for the given dataset."""
        import json

        for i in range(len(data['stock'])):  # Iterate through samples
            stock = data['stock'][i]
            day_index = data['day'][i]

            # Convert the integer index back to a string date for loading
            # Find the corresponding date in the original DataFrame
            if dataset_type == 'train':
                original_df = self.train_df
            elif dataset_type == 'validation':
                original_df = self.validation_df
            elif dataset_type == 'test':
                original_df = self.test_df
            else:
                raise ValueError(f"Invalid dataset_type: {dataset_type}")

            # Convert the integer index back to a string date
            date = str(original_df.iloc[day_index].name)

            for feature in config['data']['features']:
                # Construct paths to embedding and CI files
                embedding_file = os.path.join(config['paths']['struc2vec_dir'], dataset_type, feature, f"{stock}.json")
                ci_file = os.path.join(config['paths']['ci_dir'], dataset_type, feature, f"{stock}.json")

                # Load embedding
                if os.path.exists(embedding_file):
                    with open(embedding_file, 'r') as f:
                        embeddings = json.load(f)
                        # Get the embedding for the specific date
                        if date in embeddings:
                            data[f'{feature}_ems'] = np.array([embeddings[date][str(j)] for j in range(self.time_step)])
                        else:
                            data[f'{feature}_ems'] = None # Set to None if not found.
                else:
                    data[f'{feature}_ems'] = None

                # Load CI values
                if os.path.exists(ci_file):
                    with open(ci_file, 'r') as f:
                        ci_data = json.load(f)
                    if date in ci_data:
                        # Access CI data using string keys, and convert to float
                        ci_values = [float(ci_data[date][key]) for key in sorted(ci_data[date].keys(), key=int)]
                        data[f'{feature}_cis'] = np.array(ci_values)
                    else:
                        data[f'{feature}_cis'] = None
                else:
                    data[f'{feature}_cis'] = None
        return data

    def get_batch(self, data, start_index, batch_size):
        end_index = min(start_index + batch_size, len(data['stock']))
        batch = {}
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                if key.endswith('_ys'):
                    # The _ys data is already a 2D NumPy array (thanks to padding in dataset.py)
                    batch[key] = data[key][start_index:end_index]
                else:
                    batch[key] = data[key][start_index:end_index]
            else:
                batch[key] = data[key]  # For placeholders like _ems and _cis
        return batch


    def to_variable(self, data):
      var = []
      for i in range(self.num_features):
        feature_name = config['data']['features'][i]
        var_dict = {
            'ems': None,
            'ys': torch.tensor(data[f'{feature_name}_ys'], dtype=torch.float32).unsqueeze(-1).to(self.device),
            'cis': None
        }
        if data[f'{feature_name}_ems'] is not None:
           var_dict['ems'] = torch.tensor(data[f'{feature_name}_ems'], dtype=torch.float32).to(self.device)

        if data[f'{feature_name}_cis'] is not None:
            var_dict['cis'] = torch.tensor(data[f'{feature_name}_cis'], dtype=torch.float32).to(self.device)
        var.append(var_dict)

      return var
    
    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self.emtree.train()
            self.output.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch_idx in range(0, len(self.train_data['stock']), self.batch_size):
                with print_lock:
                    print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}/{len(self.train_data['stock']) // self.batch_size + 1}")

                batch_data = self.get_batch(self.train_data, batch_idx, self.batch_size)
                self.emtree_optim.zero_grad()
                self.output_optim.zero_grad()
                var = self.to_variable(batch_data)

                emtree_out = self.emtree(var)
                logits = self.output(emtree_out)

                targets = torch.tensor(batch_data['target'] + 1, dtype=torch.long).to(self.device)
                loss = self.loss_func(logits, targets)
                loss.backward()

                self.emtree_optim.step()
                self.output_optim.step()

                train_loss += loss.item() * len(batch_data['stock'])
                batch_predictions = torch.argmax(logits, dim=1) - 1
                train_predictions.extend(batch_predictions.cpu().detach().numpy())
                train_targets.extend(batch_data['target'])

            train_loss /= len(self.train_data['stock'])
            val_loss, val_predictions, val_targets = self.evaluate(self.validation_data)

            train_accuracy = accuracy_score(train_targets, train_predictions)
            train_precision = precision_score(train_targets, train_predictions, average='weighted', zero_division=0)
            train_recall = recall_score(train_targets, train_predictions, average='weighted', zero_division=0)
            train_f1 = f1_score(train_targets, train_predictions, average='weighted', zero_division=0)

            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_precision = precision_score(val_targets, val_predictions, average='weighted', zero_division=0)
            val_recall = recall_score(val_targets, val_predictions, average='weighted', zero_division=0)
            val_f1 = f1_score(val_targets, val_predictions, average='weighted', zero_division=0)
            with print_lock:

                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"Train Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
                print(f"Val Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'emtree_state_dict': self.emtree.state_dict(),
                    'output_state_dict': self.output.state_dict(),
                    'config': self.config,
                    }, f"{self.model_name}_best.pth")
                with print_lock:
                    print(f"Saved best model to {self.model_name}_best.pth")

    def evaluate(self, data):
        self.emtree.eval()
        self.output.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx in range(0, len(data['stock']), self.batch_size):
                batch_data = self.get_batch(data, batch_idx, self.batch_size)
                var = self.to_variable(batch_data)

                emtree_out = self.emtree(var)
                logits = self.output(emtree_out)
                targets = torch.tensor(batch_data['target'] + 1, dtype=torch.long).to(self.device)
                loss = self.loss_func(logits, targets)
                total_loss += loss.item() * len(batch_data['stock'])

                batch_predictions = torch.argmax(logits, dim=1) - 1
                all_predictions.extend(batch_predictions.cpu().numpy())
                all_targets.extend(batch_data['target'])  # get the original target

        total_loss /= len(data['stock'])
        return total_loss, all_predictions, all_targets


    def test(self):
        """Loads a trained model and evaluates it on the test set."""
        if not self.config['testing']['model_path']:
            raise ValueError("Must specify a model path in config.yaml for testing.")
        with print_lock:
            print(f"Loading model from: {self.config['testing']['model_path']}")

        checkpoint = torch.load(self.config['testing']['model_path'], map_location=self.device)
        self.emtree.load_state_dict(checkpoint['emtree_state_dict'])
        self.output.load_state_dict(checkpoint['output_state_dict'])
        self.emtree.eval()  # Set to evaluation mode
        self.output.eval()
        with print_lock:
            print("Model loaded.")

        with print_lock:
          print("Evaluating on test data...")
        test_loss, test_predictions, test_targets = self.evaluate(self.test_data)
        with print_lock:
          print("Evaluation complete.")

        test_accuracy = accuracy_score(test_targets, test_predictions)
        test_precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
        test_recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
        test_f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)
        with print_lock:
            print("\n--- Test Results ---")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Test the PriceGraph model')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    trainer = Trainer(config, args)  # Pass args to the Trainer
    if args.test:
        trainer.test()
    else:
        trainer.train()