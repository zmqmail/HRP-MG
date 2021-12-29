import torch
from torch.nn.parameter import Parameter

def test_torch_index():
    a = torch.Tensor([
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9)
    ])

    index = torch.LongTensor([
        [[1, 2],
        [0, 2]],
        [[1, 1],
        [0, 0]]
    ])

    print(torch.topk(a, 2))
    
def test_batch():
    from utils import generate_batch, sample_neighbor
    edges = [
        (1, 1, 1),
        (2, 2, 2),
        (2, 3, 4),
        (2, 4, 4),
        (3, 1, 5),
        (3, 4, 4),
    ]
    u_neigh, i_neigh = sample_neighbor(edges, 4, 5, 2, 2)
    gen = generate_batch(edges, u_neigh, i_neigh, 2)
    for batch in gen:
        print(batch)
        
def test_tensor_index():
    t = torch.Tensor([
        [1, 2],
        [2, 3],
        [3, 4]
    ])
    
    idx_t = torch.LongTensor([
        [[1,2], [0,1], [2,1], [2,2]],  
        [[2,0], [0,0], [1,1], [1,1]],  
    ])
    
    t = t[idx_t]
    print(idx_t.shape)
    print(t.shape)
    print(t)
    
def test_torch_matmul():
    
    t1 = torch.rand(10, 3, 1, 5)
    t2 = torch.rand(3, 5, 5)
    
    
    print(torch.matmul(t1, t2).shape)

def test_data_map():
    from utils import data_map, get_mapped_data
    data = [
        (2, 1, 4),
        (3, 2, 5),
        (1, 4, 2),
        (5, 4, 3)
    ]
    
    user2index, item2index, users, items = data_map(data)
    
    data.append((6, 1, 3))
    data = get_mapped_data(data, user2index, item2index)
    print(user2index)
    print(item2index)
    print(users)
    print(items)
    print(data)
    
def test_load_data():
    from utils import load_data
    data = load_data('g.txt')
    print(data)
    
def test_sample_neighbor():
    from utils import sample_neighbor
    
    networks = [
        [(1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1)],
        [(1, 3, 1), (1, 2, 1), (2, 4, 1), (3, 4, 1)],
    ]
    
    neighbors = sample_neighbor(networks, 3, 6)
    print(neighbors)
    
if __name__ == "__main__":
    test_data_map()

    
    
    
    
    