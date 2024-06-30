import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.metric import get_metric


def evaluation(harnn, batches_dt, logger, device, string='validation'):
    harnn.eval()
    with torch.no_grad():
        criterion = torch.nn.BCELoss()

        keys = ['group']
        results = {k: {} for k in keys}

        # Training loop. For each batch...
        cott = 0
        total_loss = 0
        train_loader_tqdm = tqdm(batches_dt, ncols=120)
        for idx, batch_train in enumerate(train_loader_tqdm):
            x, truths = batch_train
            gpu_x, texts, labels = [], [], []
            companies, graph_batch, weights, lengths, label_feats, text_feats, text_len, label_len, not_initial, input = x

            for elemt in input:
                gpu_x.append(elemt.to(device))

            cott += len(input)

            gpu_x = pad_sequence(gpu_x, batch_first=True, padding_value=0).to(device)

            local_logits = harnn(gpu_x, graph_batch.to(device), weights.to(device),
                                     {'text': text_feats, 'label': label_feats},
                                     {'text': text_len, 'label': label_len}, lengths, not_initial, )

            loss = 0
            truths = [truth.to(device) for truth in truths]

            for truth, pred in zip(truths[-1:], local_logits):

                loss += criterion(pred, truth)

            all_pred = [local_logits[-1]]

            total_loss += loss.cpu().data.numpy()
            train_loader_tqdm.set_description(f'{string} for the {idx}-th batch, train loss: {loss.item()}')
            for j, (truth, pred, key) in enumerate(zip(truths[-1:], all_pred, keys)):
                assert truth.shape == pred.shape
                scores = get_metric(y_true=truth.detach().cpu(), y_pred=pred.detach().cpu())
                for metric_i, metric_value in scores.items():
                    if metric_i not in results[key]:
                        results[key][metric_i] = 0
                    results[key][metric_i] += metric_value

        for k, k_items in results.items():
            for met_key, met_value in k_items.items():
                results[k][met_key] /= cott

        logger.info("Loss in {}:{}".format(string, total_loss / cott))
        logger.info("Metrics in {}:{}".format(string, results))

        return results, total_loss/cott
