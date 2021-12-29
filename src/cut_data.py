#!/user/bin/python
from numpy import random
import argparse
from utils import setup_seed, load_data

parser = argparse.ArgumentParser(description = 'cut data')
parser.add_argument('--ratio', type = float)
parser.add_argument('--seed', type = int, default=400)

args = parser.parse_args()
setup_seed(args.seed)

train_rate = args.ratio
input_path = '../yelp/ub.txt'
train_path = '../yelp/ub_{}_train.txt'.format(args.ratio)
valid_path = '../yelp/ub_{}_validate.txt'.format(args.ratio)
test_path = '../yelp/ub_{}_test.txt'.format(args.ratio)

R = load_data(input_path)

random.shuffle(R)

train_num = int(len(R) * train_rate)
test_num = len(R) - train_num
valid_num = int(train_num * 0.05)
train_num = train_num - valid_num

with open(train_path, 'w') as trainfile:
     for u, v, w in R[:train_num]:
         trainfile.write('{}\t{}\t{}\n'.format(u, v, w))

with open(valid_path, 'w') as f:
    for u, v, w in R[train_num : train_num + valid_num]:
        f.write('{}\t{}\t{}\n'.format(u, v, w))

with open(test_path, 'w') as testfile:
    for u, v, w in R[train_num + valid_num:]:
        testfile.write('{}\t{}\t{}\n'.format(u, v, w))
            


