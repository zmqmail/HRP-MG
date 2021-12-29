
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type = float)

args = parser.parse_args()

def load_graph_as_tensor(path, num1, num2) ->torch.tensor:
        t = torch.zeros(num1+1, num2+1)
        with open(path, 'r') as f:
            for line in f:
                x, y, z = line.split()
                t[int(x), int(y)] = float(z)
        return t
    

def write_homogeneous(value, index, path):
    print('Write to path: %s' % path)
    num_row, num_col = index.shape
    with open(path, 'w') as f:
        for i in range(1, num_row):
            for j in range(1, num_col):
                if value[i, j] != 0 and (i != index[i, j]):
                    f.write("{}\t{}\t{}\n".format(i, index[i, j], value[i,j]))

def generate_for_ubu(m_ub, path, topk=10):
    
    print('Generating homogeneous network for meta-path UBU')
    for r in range(1, 6):
        sub_m = torch.where(m_ub == r, torch.full_like(m_ub, 1), torch.full_like(m_ub, 0))
        if torch.cuda.is_available():
            sub_m = sub_m.cuda()
        m_tmp = torch.matmul(sub_m, torch.transpose(sub_m, 0, 1))
        m_ubu = m_tmp if r == 1 else m_ubu + m_tmp
        
    value, index = torch.topk(m_ubu, topk, 1)
    write_homogeneous(value, index, path)
    
def generate_for_ucou(uco_path, path, topk=10):
    
    print('Generating homogeneous network for meta-path UCoU')
    m_uco = load_graph_as_tensor(uco_path, 16239, 11)
    if torch.cuda.is_available():
        m_uco = m_uco.cuda()
    m_ucou = torch.matmul(m_uco, torch.transpose(m_uco, 0, 1))
    value, index = torch.topk(m_ucou, topk, 1)
    write_homogeneous(value, index, path)
    
def generate_for_ubcabu(m_ub, bca_path, path, topk=10):
    print('Generating homogeneous network for meta-path UBCaBU')
    
    m_bca = load_graph_as_tensor(bca_path, 14284, 511)
    if torch.cuda.is_available():
        m_bca = m_bca.cuda()
    for r in range(1, 6):
        sub_m_ub = torch.where(m_ub == r, torch.full_like(m_ub, 1), torch.full_like(m_ub, 0))
        if torch.cuda.is_available():
            sub_m_ub = sub_m_ub.cuda()
        m_tmp = torch.matmul(sub_m_ub, m_bca)
        m_tmp = torch.matmul(m_tmp, torch.transpose(m_tmp, 0, 1))
        m_ubcabu = m_tmp if r == 1 else m_ubcabu + m_tmp
        
    
    value, index = torch.topk(m_ubcabu, topk, 1)
    write_homogeneous(value, index, path)
    
def generate_for_ubcibu(m_ub, bci_path, path, topk=10):
    print('Generating homogeneous network for meta-path UBCiBU')
    
    m_bci = load_graph_as_tensor(bci_path, 14284, 47)
    if torch.cuda.is_available():
        m_bci = m_bci.cuda()
    for r in range(1, 6):
        sub_m_ub = torch.where(m_ub == r, torch.full_like(m_ub, 1), torch.full_like(m_ub, 0))
        if torch.cuda.is_available():
            sub_m_ub = sub_m_ub.cuda()
        m_tmp = torch.matmul(sub_m_ub, m_bci)
        m_tmp = torch.matmul(m_tmp, torch.transpose(m_tmp, 0, 1))
        m_ubcibu = m_tmp if r == 1 else m_ubcibu + m_tmp
    
    value, index = torch.topk(m_ubcibu, topk, 1)
    write_homogeneous(value, index, path)
    
def generate_for_bub(m_ub, path, topk=10):
    print('Generating homogeneous network for meta-path BUB')
    m_bu = torch.transpose(m_ub, 0, 1)
    for r in range(1, 6):
        sub_m = torch.where(m_bu == r, torch.full_like(m_bu, 1), torch.full_like(m_bu, 0))
        if torch.cuda.is_available():
            sub_m = sub_m.cuda()
        m_tmp = torch.matmul(sub_m, torch.transpose(sub_m, 0, 1))
        m_bub = m_tmp if r == 1 else m_bub + m_tmp
        
    value, index = torch.topk(m_bub, topk, 1)
    write_homogeneous(value, index, path)
    
def generate_for_bcib(bci_path, path, topk=10):
    print('Generating homogeneous network for meta-path BCiB')
    m_bci = load_graph_as_tensor(bci_path, 14284, 47)
    if torch.cuda.is_available():
        m_bci = m_bci.cuda()
    m_bcib = torch.matmul(m_bci, torch.transpose(m_bci, 0, 1)) 
    value, index = torch.topk(m_bcib, topk, 1)
    write_homogeneous(value, index, path)
    
def generate_for_bcab(bca_path, path, topk=10):
    print('Generating homogeneous network for meta-path BCaB')
    m_bca = load_graph_as_tensor(bca_path, 14284, 511)
    if torch.cuda.is_available():
        m_bca = m_bca.cuda()
    m_bcab = torch.matmul(m_bca, torch.transpose(m_bca, 0, 1)) 
    value, index = torch.topk(m_bcab, topk, 1)
    write_homogeneous(value, index, path)

if __name__ == '__main__':
    base_path = '../yelp'
    ub = "{}/ub_{}_train.txt".format(base_path, args.ratio)
    uco = "{}/uco.txt".format(base_path)
    bca = "{}/bca.txt".format(base_path)
    bci = "{}/bci.txt".format(base_path)
        
    m_ub = load_graph_as_tensor(ub, 16239, 14284)
    
    if torch.cuda.is_available():
        m_ub = m_ub.cuda()
    
    generate_for_ubu(m_ub, '{}/ubu_{}.txt'.format(base_path, args.ratio))
    generate_for_ucou(uco, '{}/ucou.txt'.format(base_path))
    generate_for_ubcabu(m_ub, bca, '{}/ubcabu_{}.txt'.format(base_path, args.ratio))
    generate_for_ubcibu(m_ub, bci, '{}/ubcibu_{}.txt'.format(base_path, args.ratio))
    generate_for_bub(m_ub, '{}/bub_{}.txt'.format(base_path, args.ratio))
    generate_for_bcab(bca, '{}/bcab.txt'.format(base_path))
    generate_for_bcib(bci, '{}/bcib.txt'.format(base_path))
                    

