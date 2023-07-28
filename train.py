# -*- coding:utf-8 -*-
import json
import os
import sys
from pathlib import Path
from Model.PatentCLS import AttentionDNY
from torch.nn.utils.rnn import pad_sequence

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
path_project = os.path.split(root_path)[0]
sys.path.append(root_path)
sys.path.append(path_project)

import time
import logging
import torch
import numpy as np
from tqdm import tqdm

import gc

from utils.data_helpers import load_adjacent

gc.collect()

from evaluation import evaluation
from utils.util import EarlyStopMonitor, set_random_seed
from utils.metric import get_metric

sys.path.append('/')

from utils import data_helpers as dh
from utils import param_parser as parser

args = parser.parameter_parser()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('./logs/{}-{}.log'.format(args.data, str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
torch.set_num_threads(1)
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.data}-{epoch}.pth'

set_random_seed(args.seed)


def save_results(path, data):
    with open(path, 'w') as file_obj:
        json.dump(data, file_obj)


if __name__ == '__main__':
    dh.tab_printer(args, logger)
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file.format(args.US))

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    logger.info("Data processing...")

    logger.info("End data processing...")

    GPU = args.gpu
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    # device_string = 'cpu'

    device = torch.device(device_string)

    batches_trains = []

    hierarchy_data, total_number = dh.load_hierarchy_tree(args.hierarchy_tree, args.US, args.hierarchy_names)
    logger.info("classification codes in hierarchical tree is:{}".format(total_number))
    train_data, company_map, companies, patent_map, patents, company_dt = \
        dh.load_data_and_labels(args, args.train_file.format(args.US), word2idx)
    print("end construct train data")
    val_data, company_map, companies, patent_map, patents, company_dt = \
        dh.load_data_and_labels(args, args.validation_file.format(args.US), word2idx,
                                company_maps=company_map, company_ids=companies,
                                patent_maps=patent_map, patent_ids=patents, company_dts=company_dt, type='val')
    print("end construct val data")
    test_data, company_map, companies, patent_map, patents, company_dt = \
        dh.load_data_and_labels(args, args.test_file.format(args.US), word2idx,
                                company_maps=company_map, company_ids=companies,
                                patent_maps=patent_map, patent_ids=patents, company_dts=company_dt, type='test')
    print("end construct test data")
    batches_train = dh.batch_iter(train_data, company_dt, total_number, args.batch_size,
                                  slide_length=args.slide_length, window=args.window, type='train', shuffle=True)
    batches_val = dh.batch_iter(val_data, company_dt, total_number, args.batch_size,
                                slide_length=args.slide_length, window=args.window, type='val', shuffle=False)
    batches_test = dh.batch_iter(test_data, company_dt, total_number, args.batch_size,
                                 slide_length=args.slide_length, window=args.window, type='test', shuffle=False)
    harnn = AttentionDNY(labels_num=total_number[2], emb_size=args.embedding_dim, hidden_size=args.lstm_dim,
                         layers_num=1, linear_size=[256, 256], dropout=args.dropout_rate, emb_init=embedding_matrix,
                         dynamic_layer=args.gcn_layer, class_nums=total_number,
                         model_way=args.model_way,
                         hierarchy_tree=hierarchy_data,
                         use_lstm_text=args.use_lstm_text,
                         dynamic_hidden=args.lstm_dim, device=device,
                         fuse_layer=args.fuse_layer).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(harnn.parameters(), lr=args.learning_rate)

    # Generate batches
    keys = ['group']
    epochs = args.epochs

    for epoch in range(epochs):
        logger.info("start epoch:{}".format(epoch))
        results = {k: {} for k in keys}
        harnn.train()
        # Training loop. For each batch...
        cott, total_loss = 0, 0
        train_loader_tqdm = tqdm(batches_train, ncols=120)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().data.numpy()

            train_loader_tqdm.set_description(f'training for the {idx}-th batch, train loss: {loss.item()}')

            for j, (truth, pred, key) in enumerate(zip(truths[-1:], local_logits, keys)):
                scores = get_metric(y_true=truth.detach().cpu(), y_pred=pred.detach().cpu(), idx=j)
                for metric_i, metric_value in scores.items():
                    if metric_i not in results[key]:
                        results[key][metric_i] = 0
                    results[key][metric_i] += metric_value

        for k, k_items in results.items():
            for met_key, met_value in k_items.items():
                results[k][met_key] /= cott
        save_results('./results/train.json', results)
        logger.info("Loss in training:{}".format(total_loss / cott))
        logger.info("Metrics in training:{}".format(results))

        print("===================================val===================================")

        val_scores, val_loss = evaluation(harnn, batches_val, logger, device,)
        save_results('./results/validate.json', val_scores)
        validate_ndcg_list = []
        for level_key, level_value in val_scores.items():
            for metric_key, metric_value in level_value.items():
                if metric_key.startswith("ndcg_"):
                    validate_ndcg_list.append(metric_value)
        validate_ndcg = np.mean(validate_ndcg_list)
        ifstop, ifimprove = early_stopper.early_stop_check(validate_ndcg)
        if ifstop:
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            harnn.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            print("===================================test===================================")
            test_scores, test_loss = evaluation(harnn, batches_test, logger, device, string='test',)
            save_results('./results/test.json', test_scores)

            exit(0)
        if ifimprove:
            logger.info("save models at epoch :{}".format(epoch))
            torch.save(
                harnn.state_dict(),
                get_checkpoint_path(early_stopper.best_epoch))

        print("===================================test===================================")
        test_scores, test_loss = evaluation(harnn, batches_test, logger, device, string='test',)
