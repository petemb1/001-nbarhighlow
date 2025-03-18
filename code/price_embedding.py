#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import json
import pickle
import numpy as np
import networkx as nx
from multiprocessing import Pool
import yaml
#Import here to prevent issues if not installed.
import pyunicorn
from pyunicorn import timeseries
import contextlib
import tempfile

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_embeddings(input_):
    ticker, feature, dataset_type, time_step = input_  # Corrected unpacking
    # Construct paths to input VG file and output embedding file
    vg_dir = os.path.join(config['paths']['vg_dir'], dataset_type, feature)
    vg_file = os.path.join(vg_dir, f'{ticker}.pickle')
    embedding_dir = os.path.join(config['paths']['struc2vec_dir'], dataset_type, feature)
    embedding_file = os.path.join(embedding_dir, f'{ticker}.json')
    os.makedirs(embedding_dir, exist_ok=True)  # Ensure output directory exists

    try:
        with open(vg_file, 'rb') as fp:
            vgs = pickle.load(fp)
    except FileNotFoundError:
        print(f"ERROR: Visibility graph file not found: {vg_file}")
        return  # Exit if the VG file doesn't exist

    embeddings = {}
    for d, adj in vgs.items():  # Iterate through dates and adjacency matrices
        labels = np.array([str(i) for i in range(len(adj))])
        G = nx.Graph()
        for i in range(len(adj)):  # Construct NetworkX graph from adjacency matrix
            vg_adjs = labels[np.where(adj[i] == 1)]
            edges = list(zip(*[[labels[i]]*len(vg_adjs), vg_adjs]))
            G.add_edges_from(edges)

        # --- Struc2Vec Embedding ---
        # Create a *unique* temporary directory for *this* Struc2Vec instance
        with tempfile.TemporaryDirectory(dir=config['paths']['temp_dir']) as temp_dir:
            #print(f"temp_dir = {temp_dir}") #debugging
            from ge import Struc2Vec  # <--- IMPORT HERE, inside the function
            model = Struc2Vec(G,
                                walk_length=config['struc2vec']['walk_length'],
                                num_walks=config['struc2vec']['num_walks'],
                                workers=1,  # Use only one worker *inside* the parallelized function.
                                verbose=0,
                                stay_prob=config['struc2vec']['stay_prob'],
                                opt1_reduce_len=config['struc2vec']['opt1_reduce_len'],
                                opt2_reduce_sim_calc=config['struc2vec']['opt2_reduce_sim_calc'],
                                opt3_num_layers=config['struc2vec']['opt3_num_layers'],
                                temp_path=temp_dir,  # Use the UNIQUE temporary directory
                                reuse=False)  # Let Struc2Vec manage its temp files.
            model.train(embed_size=config['model']['embedding_dim'],
                        window_size=config['struc2vec']['window_size'],
                        workers=1,  # Ensure only one worker here
                        iter=config['struc2vec']['iter'])
            embeddings_dict = model.get_embeddings()
            # Convert NumPy arrays to lists for JSON serialization, and stringify keys
            embeddings[str(d)] = {str(node): embedding.tolist() for node, embedding in embeddings_dict.items()}


        # --- Debug Prints (Limited) ---
        print(f"    Generated embeddings for {dataset_type}, {ticker}, {feature}, date: {d}")
        # Show the shape of the embeddings for the *first* date only.
        if len(embeddings) == 1:
           print(f"    Embedding shape: {np.array(list(embeddings[str(d)].values())).shape}")


    with open(embedding_file, 'w') as fp:
        json.dump(embeddings, fp)  # Save embeddings to JSON file
    print(f"    Saved embeddings to {embedding_file}")


def process_embeddings():

     # Load and split data
    from dataset import load_dataframe
    train_df = load_dataframe('train_df')
    validation_df = load_dataframe('validation_df')
    test_df = load_dataframe('test_df')
    dataframes = {
        'train': train_df,
        'validation': validation_df,
        'test': test_df
    }

    for dataset_type in config['vg_processing']['datasets']:  # Iterate through datasets in config
        print(f"Generating embeddings for dataset type: {dataset_type}") # Print only once per type
        if dataset_type not in dataframes:
            print(f"WARNING: Dataset type '{dataset_type}' not found. Skipping.")
            continue

        df = dataframes[dataset_type]
        tickers = df['file'].unique().tolist() # Get the unique tickers

        for ticker in tickers:
            print(f"Processing {dataset_type} data for ticker: {ticker}")
            #df_ticker = df[df['file'] == ticker]  # No longer needed
            args_list = []
            for feature in config['data']['features']:
                vg_dir = os.path.join(config['paths']['vg_dir'], dataset_type, feature)
                if not os.path.exists(vg_dir):
                    #print(f"    WARNING: VG directory not found: {vg_dir}.  Skipping feature {feature} for {ticker}.")
                    continue
                vg_file_path = os.path.join(vg_dir, f"{ticker}.pickle")

                if os.path.exists(vg_file_path):
                    args_list.append((ticker, feature, dataset_type, config['data']['time_step']))
                else:
                    print(f"WARNING: Skipping VG file (not found): {vg_file_path}")

            # Use multiprocessing: new pool for each dataset_type
            if args_list:
                with Pool() as pool:
                    pool.map(create_embeddings, args_list)
            else:
                print(f"WARNING: No VG files found for dataset type {dataset_type}, ticker {ticker}. Skipping.")

if __name__ == '__main__':
    # --- Create the base temporary directory BEFORE multiprocessing ---
    os.makedirs(config['paths']['temp_dir'], exist_ok=True)
    process_embeddings()  # No command-line arguments