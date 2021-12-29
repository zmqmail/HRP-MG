from model import *
import torch
import tqdm
from numpy import random
from utils import *

args = parse_args()
    
def train_model(network):
    
    args = parse_args()
    
    epochs = args.epoch
    batch_size = args.batch_size
    base_embedding_size = args.base_dimensions
    sub_embedding_size = args.sub_dimensions
    dim_a = args.att_dimensions
    final_embedding_size = args.final_dimensions
    num_sample = args.neighbor_samples
    patience = args.patience
    lamb = args.lamb
    
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    
    user2index, business2index, users, businesses = data_map(network)
    train_pairs = get_mapped_data(network, user2index, business2index)
    
    user_paths = ['ubu_{}'.format(args.ratio), 'ucou', 'uu']
    business_paths = ['bub_{}'.format(args.ratio), 'bcib', 'bcab']
    
    user_paths = [ '{}/{}.txt'.format('../yelp', path)  for path in user_paths]
    business_paths = [ '{}/{}.txt'.format('../yelp', path)  for path in business_paths]
    
    user_multilayer_networks = [get_mapped_data(load_data(path), user2index, user2index) for path in user_paths]
    business_multilayer_networks = [get_mapped_data(load_data(path), business2index, business2index) for path in business_paths]
    
    user_neighbors = sample_neighbor(user_multilayer_networks, num_sample, len(users))
    business_neighbors = sample_neighbor(business_multilayer_networks, num_sample, len(businesses))  

    model = HRP_MG(
        len(users), len(businesses), sub_embedding_size, \
        base_embedding_size, final_embedding_size, len(user_paths), len(business_paths), dim_a, lamb
    )
    

    model.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=1e-4
    )

    best_valid_rmse = 9999
    best_valid_mae = 9999
    best_rmse = 9999
    best_mae = 9999
    
    cur_p = 0
    for epoch in range(epochs):

        random.shuffle(train_pairs)
        batches = generate_batch(train_pairs, user_neighbors, business_neighbors, batch_size, use_gpu)

        data_iter = tqdm.tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0

        for i, data in enumerate(data_iter):
            optimizer.zero_grad()
            loss = model(*data)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))
        
        with torch.no_grad():
            valid_mae, valid_rmse =evaluate(model, get_mapped_data(valid_pairs, user2index, business2index), \
                user_neighbors, business_neighbors, use_gpu)
            
            mae, rmse = evaluate(model, get_mapped_data(test_pairs, user2index, business2index), \
                user_neighbors, business_neighbors, use_gpu)
            if valid_rmse < best_valid_rmse:
                best_valid_mae = valid_mae
                best_valid_rmse = valid_rmse
                best_mae = mae
                best_rmse = rmse
                cur_p = 0
        
            else:
                cur_p += 1
                if cur_p >= patience: 
                    print('Early stopping')
                    break
            print('validate mae:{},rmse:{}'.format(valid_mae, valid_rmse))

    print('final result mae:{}, rmse:{}'.format(best_mae, best_rmse))
    return best_mae, best_rmse


if __name__ == "__main__":
    
    train_path = '../yelp/ub_{}_train.txt'.format(args.ratio)
    test_path = '../yelp/ub_{}_test.txt'.format(args.ratio)
    valid_path = '../yelp/ub_{}_validate.txt'.format(args.ratio)
    
    train_pairs = load_data(train_path)
    test_pairs = load_data(test_path)
    valid_pairs = load_data(valid_path)
    
    train_model(train_pairs)