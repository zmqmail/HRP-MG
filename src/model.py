import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Attention(nn.Module):
    def __init__(self, dim1, dim2):
        super(Attention, self).__init__()
        self.w = Parameter(torch.FloatTensor(dim1, dim2))
        self.a = Parameter(torch.FloatTensor(dim2))
        self.b = Parameter(torch.zeros(dim2))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.w.data.uniform_(0, 0.1)
        self.a.data.uniform_(0, 0.1)

    def forward(self, node_type_embed):
        attention = F.softmax(
            torch.matmul(
                torch.tanh((torch.matmul(node_type_embed.squeeze(), self.w) + self.b)), self.a
            ).squeeze(),
            dim=1,
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed.squeeze())
        return node_type_embed

class Embedding(nn.Module):
    def __init__(
        self, num_nodes, sub_embedding_size, base_embedding_size, final_embedding_size, num_layer, dim_a, lamb):
        super(Embedding, self).__init__()
        self.num_nodes = num_nodes
        self.base_embedding_size = base_embedding_size
        self.sub_embedding_size = sub_embedding_size
        self.num_layer = num_layer
        self.dim_a = dim_a
        self.lamb = lamb
        
        self.base_embeddings = Parameter(torch.FloatTensor(num_nodes, base_embedding_size))
        self.attention_layer = Attention(sub_embedding_size, dim_a)
        self.w_1 = Parameter(torch.FloatTensor(sub_embedding_size, sub_embedding_size))
        self.w_3 = Parameter(torch.FloatTensor(sub_embedding_size + base_embedding_size, final_embedding_size))
        self.sub_embedding = Parameter(torch.FloatTensor(num_nodes, num_layer, sub_embedding_size))

        self.reset_parameters()
        

    def reset_parameters(self):
        self.base_embeddings.data.uniform_(0, 0.1)
        self.sub_embedding.data.uniform_(0, 0.1)
        self.w_1.data.uniform_(0, 0.1)
        self.w_3.data.uniform_(0, 0.1)


    def aggregate(self, node_neigh):
        node_neigh = node_neigh.type(torch.long)
        node_embed_neighbors = self.sub_embedding[node_neigh]

        node_embed_tmp = torch.cat( # tensor(Batch_size, number_layer, number_sample, sub_embedding_size)
            [
                node_embed_neighbors[:, i, :, i, :].unsqueeze(1)
                for i in range(self.num_layer)
            ],
            dim=1,
        )
        
        node_type_embed = torch.sum(node_embed_tmp, dim=2) / node_neigh.shape[2]
        return torch.relu(torch.matmul(node_type_embed.unsqueeze(2), self.w_1)).squeeze()

    def forward(self, inputs, node_neigh):
        base_embedding = self.base_embeddings[inputs]
        
        aggregatted_sub_embedding = self.aggregate(node_neigh).unsqueeze(1)
        
        integrated_embedding = self.attention_layer(aggregatted_sub_embedding)
        
        final_embedding = torch.matmul(torch.cat((base_embedding, self.lamb * integrated_embedding.squeeze()), dim=1), self.w_3)
        return torch.relu(final_embedding)

class HRP_MG(nn.Module) :
    def __init__(self, num_u, num_i, sub_embedding_size, base_embedding_size, final_embedding_size, num_layer_u,
        num_layer_i, dim_a, lamb, p=0, alpha=0.8):
        super(HRP_MG, self).__init__()

        self.num_layer_u = num_layer_u
        self.num_layer_i = num_layer_i
        
        self.user_embedding_layer = Embedding(num_u, sub_embedding_size, base_embedding_size, final_embedding_size, num_layer_u, dim_a, lamb)
        self.item_embedding_layer = Embedding(num_i, sub_embedding_size, base_embedding_size, final_embedding_size, num_layer_i, dim_a, lamb)

        self.num_u = num_u
        self.num_i = num_i
        
        self.p = p
        self.alpha = alpha

        self.mlp = nn.Sequential(nn.Linear(2 * final_embedding_size, 128), nn.Linear(128, 1))
        
        self.user_bias = Parameter(torch.zeros(num_u))
        self.item_bias = Parameter(torch.zeros(num_i))
        
    def forward(self, users, items, ratings, neigh_u, neigh_i):
        n = users.shape[0]

        pred = self.predict(users, items, neigh_u, neigh_i)
        loss = pred - ratings
        target_loss = (loss * loss).sum() + self.alpha * torch.abs(loss).sum()
        return target_loss / n

    def predict(self, users, items, neigh_u, neigh_i):
        user_embedding = self.user_embedding_layer(users, neigh_u)
        item_embedding = self.item_embedding_layer(items, neigh_i)
        # return  torch.matmul(user_embedding.unsqueeze(1), item_embedding.unsqueeze(2)).squeeze() + \
        #   + self.p * (self.user_bias[users] + self.item_bias[items])
        
        # mlp has a better performance
        return  self.mlp(torch.cat((user_embedding, item_embedding), dim=1)).squeeze() \
            + self.p * (self.user_bias[users] + self.item_bias[items])
        
        
        