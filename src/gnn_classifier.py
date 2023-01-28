import time

import numpy as np
import torch
import torch.nn.functional as F
from scent_data import LeffingwellGoodscentsDataset
from torch import nn, optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, global_add_pool

# import tqdm


class GNNScentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_passing_layers = GatedGraphConv(out_channels=43,
                                                     num_layers=5,
                                                     aggr='add')
        self.ff1 = nn.Linear(43, 392)
        # self.batch_norm_1 = nn.BatchNorm1d(392)
        self.ff2 = nn.Linear(392, 392)
        # self.batch_norm_2 = nn.BatchNorm1d(392)
        self.ff3 = nn.Linear(392, 392)
        # self.batch_norm_3 = nn.BatchNorm1d(392)

        self.pred_layer = nn.Linear(392, 138)
    
    def forward(self, data, minibatch=True):
        x, edge_index = data.x, data.edge_index

        x = x.to(torch.float)
        x = self.message_passing_layers(x, edge_index)
        x = global_add_pool(x, batch=data.batch if minibatch else None)

        # for dense_layer, batch_norm in zip([self.ff1, self.ff2, self.ff3], [self.batch_norm_1, self.batch_norm_2, self.batch_norm_3]):
        for dense_layer in [self.ff1, self.ff2, self.ff3]:
            x = dense_layer(x)
            x = F.relu(x)
            # if minibatch:
            #     x = batch_norm(x)
            x = F.dropout(x, p=0.12, training=self.training)
       
        x = self.pred_layer(x)
        final_activation = nn.Sigmoid()
        x = final_activation(x)
        return x

def main():
    # Load vendi weights:
    with open('vendi_cache.npy', 'rb') as f:
        vendi_weights = np.load(f)
        vendi_weights = torch.tensor(vendi_weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_dataset_loading = time.time()
    dataset = LeffingwellGoodscentsDataset(verbose=False)
    print(f"Finish preparing the database; it takes {time.time() - start_dataset_loading} seconds")
    
    proportions = [.8, .2]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_data, valid_data = random_split(dataset, lengths)
        
    model = GNNScentClassifier().to(device).float()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=.05)
    
    train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=3, shuffle=False)
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        input, target = batch[0], batch[1]
        optimizer.zero_grad()
        output = model(input)
        final_loss = nn.BCELoss(weight=vendi_weights)
        # final_loss = F.cross_entropy()
        target = target.to(device).float() # For some reason target is in float64, not float32
        loss = final_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % 10000:
            total_valid_loss = 0
            with torch.inference_mode():
                for j, valid_batch in enumerate(valid_loader):
                    valid_input, valid_target = valid_batch[0], valid_batch[1]
                    valid_output = model(valid_input)
                    # valid_target_batched = valid_target.unsqueeze(0).float()
                    valid_target = valid_target.to(device).float()
                    total_valid_loss += final_loss(valid_output, valid_target)
            print(f'Train loss is {loss} \t Valid loss is {total_valid_loss/(j+1)} \t Time elapsed is {time.time() - start_time}.')

    # for i, (input, target) in enumerate(train_data):
    #     optimizer.zero_grad()
    #     output = model(input)
    #     target_batched = target.unsqueeze(0).float()
    #     final_loss = nn.BCELoss()
    #     # final_loss = F.cross_entropy
    #     loss = final_loss(output, target_batched)
    #     loss.backward()
    #     optimizer.step()
    #     if i % 10000:
    #         total_valid_loss = 0
    #         with torch.inference_mode():
    #             for j, (valid_input, valid_target) in enumerate(valid_data):
    #                 valid_output = model(valid_input)
    #                 valid_target_batched = valid_target.unsqueeze(0).float()
    #                 total_valid_loss += final_loss(valid_output, valid_target_batched)
    #         print(f'Valid loss is {total_valid_loss/len(valid_data)}')


if __name__ == '__main__':
    main()