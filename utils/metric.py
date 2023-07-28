import numpy as np
import torch
from tqdm import tqdm
import datetime


def recall_score(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=8)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    # print(f'recall {top_k} predict_values: {value[8:11]}')
    # print(f'recall {top_k} predict_indices: {predict_indices[8:11]}')
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = (torch.logical_and(torch.logical_and(predict, truth), truth == 1)).sum(-1), truth.sum(-1)
    # print(f'recall {top_k}, tp: {tp[8:11]}, t: {t[8:11]}, ')
    # end_time = datetime.datetime.now()
    # print("recall_score cost %d seconds" % (end_time - start_time).seconds)
    return (tp.float() / t.float()).sum().item()


def precision(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=8)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = (torch.logical_and(torch.logical_and(predict, truth), truth == 1)).sum(-1), predict.sum(-1)

    return (tp.float() / t.float()).sum().item()


def F1_scr(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=8)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, p, t = (torch.logical_and(torch.logical_and(predict, truth), truth == 1)).sum(-1), predict.sum(-1), truth.sum(
        -1)
    recall = tp.float() / t.float()
    prec = tp.float() / p.float()

    eps = 1e-6
    f1 = 2 * recall * prec / (recall + prec + eps)

    return recall.sum().item(), prec.sum().item(), f1.sum().item(), recall, prec


def dcg(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):

    Returns:

    """
    value, predict_indices = y_pred.topk(k=8)
    predict_indices = predict_indices[:, :top_k]
    gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)  # (batch_size,)


def ndcg_score(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):
    Returns:

    """
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)

    return (dcg_score / idcg_score).sum().item()


def PHR(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=8)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    hit_num = torch.logical_and(predict, truth).sum(dim=1).nonzero(as_tuple=False).shape[0]

    # return hit_num / truth.shape[0]
    return hit_num


def get_metric(y_true, y_pred, idx=0):
    """
        Args:
            y_true: tensor (samples_num, items_total)
            y_pred: tensor (samples_num, items_total)
        Returns:
            scores: dict
    """
    # idx_list = [[1],[1,3,5],[1,3,5],[1,3,5]]
    # idx_list = [[1], [1, 3, 5], [1, 3, 5]]
    idx_list = [[1, 3, 5]]
    result = {}
    for i, top_k in enumerate(idx_list[idx]):
        recall, prec, F1, rec_list, prec_list = F1_scr(y_true, y_pred, top_k=top_k)
        result.update({
            f'precision_{top_k}': prec,
            f'recall_{top_k}': recall,
            f'F1_{top_k}': F1,
            f'ndcg_{top_k}': ndcg_score(y_true, y_pred, top_k=top_k),
            f'PHR_{top_k}': PHR(y_true, y_pred, top_k=top_k)
        })
    return result


''' 
def evaluate(model, data_loader):
    """
    Args:
        model: nn.Module
        data_loader: DataLoader
    Returns:
        scores: dict
    """
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                tqdm(data_loader)):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            predict_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

            # predict_data shape (batch_size, baskets_num, items_total)
            # truth_data shape (batch_size, baskets_num, items_total)
            y_pred.append(predict_data.detach().cpu())
            y_true.append(truth_data.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        return get_metric(y_true=y_true, y_pred=y_pred)
'''
