import argparse
import numpy as np
import torch
import math
from numpy import random

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lamb', type=float, default=0.5)

    parser.add_argument('--ratio', type=float,
                        help='Input dataset path')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')
    
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Number of batch_size. Default is 64.')
    
    parser.add_argument('--base-dimensions', type=int, default=100,
                        help='Number of base dimensions. Default is 100.')
    
    parser.add_argument('--sub-dimensions', type=int, default=40,
                        help='Number of sub dimensions. Default is 400.')
    
    parser.add_argument('--att-dimensions', type=int, default=100,
                        help='Number of attention dimensions. Default is 100.')
    
    parser.add_argument('--final-dimensions', type=int, default=200,
                        help='Number of final dimensions. Default is 200.')

    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    
    
    return parser.parse_args()
                
def load_data(path) :
    data = list()
    with open(path, 'r') as f:
        for line in f:
            u, v, w = line.strip().split()
            data.append((int(u), int(v), float(w)))
    return data

def data_map(data):
    '''
    Map data to proper indices in case they are not in a continues [0, N) range
    
    Parameters
    ----------
    data: list of tuples, [(user, item, rating), ...]
    
    Returns
    -------
    user2index: {user: index}
    item2index: {item: index}
    index2user: [user]
    index2item: [item]
    '''
    
    users = sorted(list(set([user for user, _, _ in data])))
    items = sorted(list(set([items for _, items, _ in data])))
    
    user2index = {user: index for index, user in enumerate(users)}
    item2index = {item: index for index, item in enumerate(items)}
    
    return user2index, item2index, users, items

def get_mapped_data(data, dic1, dic2):
    mapped_data = list()
    for u, v, w in data:
        try:
            mapped_data.append((dic1[u], dic2[v], w))
        except: ...
    return mapped_data

def sample_neighbor(networks, num_sample, num_nodes) :
    '''
    Samples neighbors for nodes in each layer of networks
    
    Parameters
    ----------
    networks: list, [[(u1, v1, w1), (u2, v2, w2), ...], [(), ...], ...]
    num_sample: int
    num_nodes: int
    
    Returns
    -------
    neighbors: list
    '''
    num_layer = len(networks)
    
    neighbors = [[[] for _ in range(num_layer)] for i in range(num_nodes)]
    
    for layer_idx, network in enumerate(networks):
        for u, v, e in network:
            neighbors[u][layer_idx].append(v)
            
    for layer_idx in range(num_layer):
        for node_idx in range(num_nodes):
            num_neighbors = len(neighbors[node_idx][layer_idx])
            if num_neighbors == 0:
                neighbors[node_idx][layer_idx] = [node_idx] * num_sample
            
            elif num_neighbors < num_sample:
                neighbors[node_idx][layer_idx].extend(
                    list(
                        np.random.choice(
                            neighbors[node_idx][layer_idx],
                            size=num_sample - num_neighbors
                            )
                        )
                    )
            
            elif num_neighbors > num_sample:
                neighbors[node_idx][layer_idx] = list(
                    np.random.choice(neighbors[node_idx][layer_idx], size=num_sample, replace=False)
                )
        
    return neighbors

def generate_batch(edges, user_neighbors, item_neighbors, batch_size, use_gpu=False):
    '''
    Geneates training batches
    
    Params
    ------
    edges: list of tuples, [(user, item, rating), ...]
    user_neigbors/item_neighbors: list
    batch_size: int
    use_gpu: bool
    
    Returns
    -------
    batch_user_tensor/batch_item_tensor/batch_rating_tensor: tensor(batch_size)
    batch_user_neighbors_tensor/batch_item_neighbors_tensor: tensor(batch_size, num_nodes, num_layer, num_sample)
    '''
    random.shuffle(edges)
    edges_tensor = torch.LongTensor(edges)
    user_neighbors_tensor = torch.LongTensor(user_neighbors)
    item_neighbors_tensor = torch.LongTensor(item_neighbors)
    
    if use_gpu:
        edges_tensor = edges_tensor.cuda()
        user_neighbors_tensor = user_neighbors_tensor.cuda()
        item_neighbors_tensor = item_neighbors_tensor.cuda()
        
    num_batch = (len(edges) + batch_size) // batch_size
    for i in range(num_batch):
        batch_edges_tensor = edges_tensor[i*batch_size : (i+1) * batch_size] \
            if i != num_batch - 1 else edges_tensor[i*batch_size : len(edges_tensor)]
            
        batch_user_tensor = batch_edges_tensor[:, 0]
        batch_item_tensor = batch_edges_tensor[:, 1]
        batch_rating_tensor = batch_edges_tensor[:, 2]
        
        batch_user_neighbors_tensor = user_neighbors_tensor[batch_user_tensor]
        batch_item_neighbors_tensor = item_neighbors_tensor[batch_item_tensor]
        
        yield batch_user_tensor, \
            batch_item_tensor, \
            batch_rating_tensor, \
            batch_user_neighbors_tensor, \
            batch_item_neighbors_tensor
    
def evaluate(model, test_data, user_neighbors, item_neighbors, use_gpu=False, batch_size=10000):
        mae = 0
        rmse = 0
        for users, items, ratings, u_neigh, i_neigh \
            in generate_batch(test_data, user_neighbors, item_neighbors, batch_size, use_gpu):
                
            pred = model.predict(users, items, u_neigh, i_neigh).clamp(1, 5)
            dif = pred - ratings
            mae += abs(dif).sum()
            rmse += (dif * dif).sum()
        n = len(test_data)
        return float(mae) / n, math.sqrt(rmse / n)
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True 