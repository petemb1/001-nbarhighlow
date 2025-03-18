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
import contextlib

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def vg_ci(_input):
    file_, feature, time_step, dataset_type = _input  # Include dataset_type
    graph_file = os.path.join(config['paths']['vg_dir'], dataset_type, feature, file_ +'.pickle') # Use dataset type

    try:
        with open(graph_file, 'rb') as fp:
            vgs = pickle.load(fp)
    except FileNotFoundError:
        print(f"ERROR: Visibility graph file not found: {graph_file}")
        return  # Exit if the VG file doesn't exist

    cis = {}
    num_dates_visualized = 0
    for d, adj in vgs.items():
        labels = np.array([str(i) for i in range(time_step)])
        G = nx.Graph()
        for i in range(time_step):
            vg_adjs = labels[np.where(adj[i] == 1)]
            edges = list(zip(*[[labels[i]]*len(vg_adjs), vg_adjs]))
            G.add_edges_from(edges)
        #Import here to prevent issues if not installed.
        from lib.ci import collective_influence
        cis[d] = collective_influence(G) # Calculate CI

        # --- ADDED DEBUG PRINTS ---
        if num_dates_visualized < 3: # Limit number of printouts.
          print(f"    CI values (first 5) for {dataset_type}, {file_}, {feature}, date {d}:")
          # Convert to a list of items and print first 5
          items = list(cis[d].items())
          for node, ci_value in items[:5]:
              print(f"      Node: {node}, CI: {ci_value}")
          print(f"    Shape of CI data for this date: {np.array(list(cis[d].values())).shape}")  # Check shape
          num_dates_visualized += 1
        # --- END DEBUG PRINTS ---

    ci_dir = os.path.join(config['paths']['ci_dir'], dataset_type, feature) # Use dataset_type
    os.makedirs(ci_dir, exist_ok=True) #Create correct output location.
    ci_file = os.path.join(ci_dir, '%s.json' % file_)
    with open(ci_file, 'w') as fp:
        json.dump(cis, fp)
    print(f"    Saved CI data to {ci_file}") # Show where file is saved


def process_ci():
    # No more data loading here!

    for dataset_type in config['vg_processing']['datasets']: # Use the datasets setting in the config file.
        print(f"Calculating Collective Influence for dataset type: {dataset_type}")

        # We don't have direct access to the dataframes here, so we need to
        # infer the tickers from the directory structure.  This is slightly
        # less elegant than iterating through dataframes, but it's necessary
        # to avoid redundant loading.

        vg_base_dir = config['paths']['vg_dir']
        dataset_vg_dir = os.path.join(vg_base_dir, dataset_type)

        if not os.path.exists(dataset_vg_dir):
            print(f"WARNING: VG directory for dataset type '{dataset_type}' not found. Skipping.")
            continue

        tickers = set()
        for feature in config['data']['features']:
            feature_dir = os.path.join(dataset_vg_dir, feature)
            if os.path.exists(feature_dir):
                for filename in os.listdir(feature_dir):
                    if filename.endswith(".pickle"):
                        tickers.add(filename[:-7])  # Extract ticker from filename

        for ticker in tickers:
          print(f"Processing {dataset_type} data for ticker: {ticker}")
          args_list = []
          for feature in config['data']['features']:  # Iterate through features
                vg_dir = os.path.join(config['paths']['vg_dir'], dataset_type, feature)
                if not os.path.exists(vg_dir):
                    #print(f"    WARNING: VG directory not found: {vg_dir}.  Skipping feature {feature} for {ticker}.")
                    continue #Skip this feature

                # Construct full path to VG pickle file
                vg_file_path = os.path.join(vg_dir, f"{ticker}.pickle")
                if os.path.exists(vg_file_path):  # Check if the VG file exists
                    args_list.append((ticker, feature, config['data']['time_step'], dataset_type))
                else:
                     print(f"WARNING: Skipping VG file (not found): {vg_file_path}")

          # Use multiprocessing
          if args_list:  # Only run if there are valid files
            with Pool() as pool:
                pool.map(vg_ci, args_list)
          else:
            print(f"WARNING: No VG files found for dataset type {dataset_type}, ticker {ticker}. Skipping.")

if __name__ == '__main__':
    process_ci() # No command-line arguments