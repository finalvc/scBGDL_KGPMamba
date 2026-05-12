import pandas as pd
import numpy as np
import torch
import argparse
import os
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from pathlib import Path
from model import GDL
from utils.dataset import scBdata
from utils.util import make_batch, collate, CoxLoss, weight_init
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Set up the CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser(description='Graph-based deep learning')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--workspace', default='../Datasets/Survival_data/{cohort}_cohort.csv')
    parser.add_argument('--ScMatrix', default='../Datasets/Single-Cell Expression/{cohort}_matrix.csv')
    parser.add_argument('--nodes', default='../Datasets/graphs/{cohort}_nodes.npy')
    parser.add_argument('--edges', default='../Datasets/graphs/{cohort}_edges.npy')
    parser.add_argument('--checkpoint-dir', default='../checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--wd', default=5e-5)
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    """Train the graph-based deep learning model."""
    args = get_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Load dataset
    workspace = pd.read_csv(args.workspace)
    ScMatrix = pd.read_csv(args.ScMatrix)
    scaler = StandardScaler()
    ScMatrix.iloc[:, 1:] = scaler.fit_transform(ScMatrix.iloc[:, 1:])
    n_genes = ScMatrix.shape[1] - 1  # Number of genes (exclude the ID column)
    # Initialize model
    model = GDL(input_dim=n_genes).cuda()
    model.apply(weight_init)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=200)
    checkpoint_path = args.checkpoint_dir / f'best_model_{seed}.pth'

    # Early stopping mechanism
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)

    # Prepare data loader
    data_TCGA = scBdata(workspace, args.nodes, args.edges)
    data_TCGA_loader = DataLoader(data_TCGA, args.batch_size, shuffle=True, num_workers=0, drop_last=False, collate_fn=collate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        print(datetime.now())
        for graphs in data_TCGA_loader:
            batch_graphs, batch_OSs, batch_OS_times, batch_slides_name = make_batch(graphs)
            optimizer.zero_grad()
            OCDPI, _ = model(batch_graphs)
            loss = CoxLoss(batch_OSs.cuda(), batch_OS_times.cuda(), OCDPI)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        epoch_loss_total = np.sum(epoch_loss)
        print(f'Epoch {epoch} total loss: {epoch_loss_total}')

        # Early stopping check
        early_stopping(epoch_loss_total, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

    # Save the trained model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

