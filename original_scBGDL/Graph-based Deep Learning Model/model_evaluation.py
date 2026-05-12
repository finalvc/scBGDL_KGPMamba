import pandas as pd
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from utils.dataset import scBdata
from utils.util import make_batch, collate
import os

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Graph-based deep learning model')
    parser.add_argument('--workspace', default='../Datasets/Survival_data/{cohort}_cohort.csv')
    parser.add_argument('--nodes', default='../Datasets/graphs/{cohort}_nodes.npy')
    parser.add_argument('--edges', default='../Datasets/graphs/{cohort}_edges.npy')
    parser.add_argument('--checkpoint-dir', default='../checkpoint/', type=str, metavar='DIR', help='Path to checkpoint directory')
    args = parser.parse_args()
    return args

def evaluation(model, cohort=''):
    """Evaluate the model and compute risk scores for each clinical sample."""
    workspace = pd.read_csv(f'../Datasets/Survival_data/{cohort}_cohort.csv')
    nodes_path = f'../Datasets/graphs/{cohort}_nodes.npy'
    edges_path = f'../Datasets/graphs/{cohort}_edges.npy'

    data = scBdata(workspace, nodes_path, edges_path)
    data_loader = DataLoader(data, 8, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate)

    report = pd.DataFrame()
    slides_name = []
    OCDPIs = np.array([])  # Risk scores
    OS_times = np.array([])  # Survival time
    OSs = np.array([])  # Survival status

    with torch.no_grad():
        for graphs in data_loader:
            batch_graphs, batch_OSs, batch_OS_times, batch_slides_name = make_batch(graphs)
            batch_OCDPI, _ = model(batch_graphs)
            OCDPIs = np.append(OCDPIs, batch_OCDPI.detach().cpu().numpy())
            slides_name += batch_slides_name
            OS_times = np.append(OS_times, batch_OS_times)
            OSs = np.append(OSs, batch_OSs.detach().cpu().numpy())

    report['slides'] = slides_name
    report['OS.time'] = OS_times
    report['OS'] = OSs
    report['OCDPI'] = OCDPIs
    return report

