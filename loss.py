import torch
from geomloss import SamplesLoss

def batch_pairwise_dist(x, y):
    # 32, 2500, 3
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    # print(bs, num_points, points_dim)
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x).type(torch.cuda.LongTensor)
    diag_ind_y = torch.arange(0, num_points_y).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


def batch_NN_loss(x, y):
    # bs, num_points, points_dim = x.size()
    # dist1 = torch.sqrt(batch_pairwise_dist(x, y))
    # values1, indices1 = dist1.min(dim=2)
    #
    # dist2 = torch.sqrt(batch_pairwise_dist(y, x))
    # values2, indices2 = dist2.min(dim=2)
    # a = torch.div(torch.sum(values1,1), num_points)
    # b = torch.div(torch.sum(values2,1), num_points)
    # sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)
    # return sum

    P = batch_pairwise_dist(x, y)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.mean(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.mean(mins)
    return loss_1 + loss_2

def batch_EMD_loss(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    batch_EMD = 0
    L = SamplesLoss()
    for i in range(bs):
        loss = L(x[i], y[i])
        batch_EMD += loss
    emd = batch_EMD/bs
    return emd