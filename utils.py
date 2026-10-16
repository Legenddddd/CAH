import numpy as np
import torch
from model import Model
from Resnet import resnet18, resnet50
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

class Hash_Loss_PC(nn.Module):
    def __init__(self, code_length):
        super(Hash_Loss_PC, self).__init__()
        self.code_length = code_length

    def forward(self, F, C, S_C, B, S_B):
        hash_loss_pair = ((self.code_length * S_B - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0])
        hash_loss_center = ((self.code_length * S_C - F @ C.t()) ** 2).sum() / (F.shape[0] * C.shape[0])
        loss = hash_loss_pair + hash_loss_center
        return loss



def dataset_label(dataloader, classes):
    for i, (index, img, labels) in enumerate(dataloader):
        if i == 0:
            train_labels = labels
            train_indexs = index
        else:
            train_labels = torch.cat((train_labels, labels))
            train_indexs = torch.cat((train_indexs, index))

    pair_matrix = (train_labels[train_indexs].unsqueeze(1) == train_labels[train_indexs].unsqueeze(0))
    L_tr = one_hot_label(train_labels.cpu(), classes).numpy()

    return L_tr, pair_matrix


def load_model(backbone, pretrain=True, require_grad=True, components_num=200, topn=4, bit=16, alpha=20):
    print('Building model..', bit)
    feature_size = 512
    if backbone == 'resnet50':
        num_ftrs = 2048
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = Model(net, feature_size, num_ftrs, components_num, topn=topn, bit=bit, alpha=alpha)
    elif backbone == 'resnet18':
        num_ftrs = 512
        net = resnet18(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = Model(net, feature_size, num_ftrs, components_num, topn=topn, bit=bit, alpha=alpha)

    return net


def calc_sim(L1, L2):
    S = (L1 @ L2.t() > 0).float()
    # soft constraint
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S


def one_hot_label(label,num_class):
    label = label.view(-1,1).long().cpu()
    onehot = torch.zeros(label.shape[0],num_class)
    onehot = onehot.scatter(1,label,1)
    return onehot


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
        print('retrieval size:', retrieval_L.shape[0])
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.int)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd, as_tuple=False)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def count_components(pair_matrix):
    adj_np = pair_matrix.cpu().numpy()
    adj_sparse = csr_matrix(adj_np)
    n, _ = connected_components(csgraph = adj_sparse, directed = False, return_labels = True)
    return n