# -*- coding:utf-8 -*-
import os
import time
import heapq

import dgl
import logging
import json

import jieba
import numpy as np
from collections import OrderedDict, Counter

import pandas as pd
from texttable import Texttable
from gensim.models import word2vec, KeyedVectors
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def _option(pattern):
    """
    Get the option according to the pattern.
    pattern 0: Choose training or restore.
    pattern 1: Choose best or latest checkpoint.

    Args:
        pattern: 0 for training step. 1 for testing step.
    Returns:
        The OPTION.
    """
    if pattern == 0:
        OPTION = input("[Input] Train or Restore? (T/R): ")
        while not (OPTION.upper() in ['T', 'R']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    if pattern == 1:
        OPTION = input("Load Best or Latest Model? (B/L): ")
        while not (OPTION.isalpha() and OPTION.upper() in ['B', 'L']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger.
        input_file: The logger file path.
        level: The logger level.
    Returns:
        The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_out_dir(option, logger):
    """
    Get the out dir for saving model checkpoints.

    Args:
        option: Train or Restore.
        logger: The logger.
    Returns:
        The output dir for model checkpoints.
    """
    if option == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {0}\n".format(out_dir))
    if option == 'R':
        MODEL = input("[Input] Please input the checkpoints model you want to restore, "
                      "it should be like (1490175368): ")  # The model you want to restore

        while not (MODEL.isdigit() and len(MODEL) == 10):
            MODEL = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
        logger.info("Writing to {0}\n".format(out_dir))
    return out_dir


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name.
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(output_file, data_id, true_labels, predict_labels, predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network.
        data_id: The data record id info provided by dict <Data>.
        true_labels: The all true labels.
        predict_labels: The all predict labels by threshold.
        predict_scores: The all predict scores by threshold.
    Raises:
        IOError: If the prediction file is not a .json file.
    """
    if not output_file.endswith('.json'):
        raise IOError("[Error] The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(predict_labels)
        for i in range(data_size):
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', [int(i) for i in true_labels[i]]),
                ('predict_labels', [int(i) for i in predict_labels[i]]),
                ('predict_scores', [round(i, 4) for i in predict_scores[i]])
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted one-hot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted one-hot labels based on the topK.

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_labels: The predicted labels.
        predicted_scores: The predicted scores.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        The predicted labels.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def create_metadata_file(word2vec_file, output_file):
    """
    Create the metadata file based on the corpus file (Used for the Embedding Visualization later).

    Args:
        word2vec_file: The word2vec file.
        output_file: The metadata file path.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist.")

    wv = KeyedVectors.load(word2vec_file, mmap='r')
    word2idx = dict([(k, v.index) for k, v in wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("[Warning] Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def load_word2vec_matrix(word2vec_file):
    """
    Get the word2idx dict and embedding matrix.

    Args:
        word2vec_file: The word2vec file.
    Returns:
        word2idx: The word2idx dict.
        embedding_matrix: The word2vec model matrix.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    model = word2vec.Word2Vec.load(word2vec_file, mmap='r')
    print(model)
    word2idx = OrderedDict({"PAD_": 0, "_UNK": 1})
    embedding_size = model.wv.vector_size
    count = 2
    for k, v in model.wv.vocab.items():
        word2idx[k] = count
        # print(k, count)
        count += 1
    vocab_size = len(word2idx)
    print(len(model.wv.vocab.keys()), vocab_size)
    embedding_matrix = torch.zeros([vocab_size, embedding_size])
    for key, value in word2idx.items():
        if key == 'PAD_':
            continue
        elif key == "_UNK":
            embedding_matrix[value] = torch.randn(embedding_size)
        else:
            embedding_matrix[value] = torch.tensor(model.wv[key])
    return word2idx, embedding_matrix


def load_data_and_labels(args, input_file, word2idx: dict, company_maps=None, company_ids=0,
                         patent_maps=None, patent_ids=1, company_dts=None, type='train'):
    """
    Load research data from files, padding sentences and generate one-hot labels.

    Args:
        args: The arguments.
        input_file: The research record.
        word2idx: The word2idx dict.
    Returns:
        The dict <Data> (includes the record tokenindex and record labels)
    Raises:
        IOError: If word2vec model file doesn't exist
    """

    def _token_to_index(x: list):
        result = []
        for item in x:
            if item not in word2idx.keys():
                result.append(word2idx['_UNK'])
                # result.append(word2idx['<UNK>'])
            else:
                word_idx = word2idx[item]
                result.append(word_idx)
        return torch.tensor(result, dtype=torch.long)

    Data = dict()

    company_id = company_ids
    company_map = company_maps if company_maps else dict()

    patent_id = patent_ids
    patent_map = patent_maps if patent_maps else dict()

    company_dt = company_dts if company_dts else {'patents': {}, 'company': {}}

    fin = pd.read_csv(input_file)
    fin.sort_values(by='publicationDate', inplace=True)
    Data['content_index'] = []
    Data['section'] = []
    Data['subsection'] = []
    Data['group'] = []
    # Data['subgroup'] = []
    Data['company'] = []
    Data['patentId'] = []

    for id, patent in tqdm(fin.iterrows()):

        if patent['orgID'] not in company_map:
            if type != 'train':
                print(patent['orgID'])
                continue
            else:
                company_map[patent['orgID']] = company_id
                company_id += 1

        if args.US == 'US':
            content = patent['summary'].split()[:100]
            # content = sorted(random.sample(content, min(100, len(content))))
        else:
            content = (" ".join(jieba.cut(patent['summary'], cut_all=False))).split()[:100]
            # content = sorted(random.sample(content, min(60, len(content))))
        section = list(map(int, patent['codeIId'].split(";")))
        subsection = list(map(int, patent['codeIIId'].split(";")))
        group = list(map(int, patent['codeIIIId'].split(";")))
        subgroup = list(map(int, patent['codeIVId'].split(";")))

        content_id = _token_to_index(content)
        Data['content_index'].append(content_id)
        Data['section'].append(section)
        Data['subsection'].append(subsection)
        Data['group'].append(group)
        # Data['subgroup'].append(subgroup)

        Data['company'].append(company_map[patent['orgID']])

        if patent['patentId'] not in patent_map:
            patent_map[patent['patentId']] = patent_id
            patent_id += 1
        Data['patentId'].append(patent_map[patent['patentId']])

        if True:
            if company_map[patent['orgID']] not in company_dt['company']:
                company_dt['company'][company_map[patent['orgID']]] = {"pid": [], "text": [], "label": [],
                                                                       'text_len': [],
                                                                       'label_len': []}
            company_dt['patents'][patent_map[patent['patentId']]] = len(
                company_dt['company'][company_map[patent['orgID']]]["pid"])
            company_dt['company'][company_map[patent['orgID']]]['pid'].append(patent_map[patent['patentId']])
            company_dt['company'][company_map[patent['orgID']]]['text'].append(content_id)
            company_dt['company'][company_map[patent['orgID']]]['text_len'].append(len(content_id))
            company_dt['company'][company_map[patent['orgID']]]['label_len'].append(len(group))
            company_dt['company'][company_map[patent['orgID']]]['label'].append(torch.tensor(group, dtype=torch.long))

    return Data, company_map, company_id, patent_map, patent_id, company_dt


def collate_fn(data):
    """
    Args:
        data (List[Tuple]): len(data) = batch_size
            tuple[0] (int): user_id
            tuple[1] (Tensor): items, shape (num_items,)
    Returns:
        users (Tensor): shape (batch_size,)
        targets (Tensor): shape (batch_size, items_total)
    """
    input, truth = [], []
    for i, items in enumerate(zip(*data)):
        if i > 2:
            items = torch.cat(items, dim=0)
            truth.append(items)
        elif i == 1:
            for idx, item in enumerate(zip(*items)):
                # print(type(item))
                if isinstance(item[0], dgl.DGLGraph):
                    input.append(dgl.batch(item))
                elif isinstance(item[0], torch.Tensor):
                    if idx == 1:
                        edges_weight, lengths = list(), list()
                        for data in item:
                            edges_weight.append(data.flatten())
                            assert data.size(0) != 0
                            lengths.append(data.size(0))
                        # (T_max, N_1*N_1 + N_2*N_2 + ... + N_b*N_b)
                        input.append(torch.cat(edges_weight))
                        input.append(lengths)
                    elif idx < 6:
                        input.append(item)
                    else:
                        input.append(torch.cat(item))
                else:
                    item_list = []
                    for i_item in item:
                        item_list.extend(i_item)
                    input.append(item_list)
        else:
            input.append(items)
    return input, truth


class My_dataset(Dataset):
    def __init__(self, data, company_dts, total_num, slide_length, window, type='train'):
        super().__init__()
        self.data = data
        self.slide_length = slide_length
        self.window = window
        # self.keys = ['content_index', 'company', 'section', 'subsection', 'group', 'subgroup']
        self.keys = ['content_index', 'section', 'subsection', 'group']
        self.total_num = total_num
        self.length = len(self.data[self.keys[0]])
        self.companies = company_dts  # "patents":(idx, text, group); "company":[id1, id2, ...]
        self.graphs, self.weights = self.get_Graphs(115)
        self.type = type

    def get_Graphs(self, max_number):
        graphs, weights = [], []
        for i in range(1, max_number):
            project_nodes = torch.tensor(list(range(i)))
            src = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=1).flatten().tolist()
            dst = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=0).flatten().tolist()
            g = dgl.graph((src, dst), num_nodes=project_nodes.shape[0])
            edges_weight = self.get_edges_weight(project_nodes, self.window)
            graphs.append(g)
            weights.append(edges_weight)
        return graphs, weights

    def _create_onehot_labels(self, labels_index, num_labels):
        label = torch.zeros([num_labels])
        # print(labels_index)
        if isinstance(labels_index, torch.Tensor):
            label[labels_index] = 1
        else:
            index_lb = torch.tensor(labels_index, dtype=torch.long)
            label[index_lb] = 1
        return label.unsqueeze(dim=0)

    def __getitem__(self, index):
        # print("fetch index:{}".format(index))
        item_dt = []
        company = self.data['company'][index]
        pid = self.data['patentId'][index]
        if True:
            index_com = self.companies['patents'][pid]
        # else:
        #     index_com = len(self.companies['company'][company]['pid'])
        not_initial = torch.ones(1, dtype=torch.long)
        if index_com == 0:
            not_initial[0] = 0
            nodes = [0]
            text_len = [1]
            label_len = [1]
            text_nodes = [torch.tensor([0])]
            label_nodes = [torch.tensor([0])]
        else:
            # start = max(index_com - 60, 0)
            start = max(index_com - self.slide_length, 0)
            nodes = self.companies['company'][company]['pid'][start:index_com]
            text_nodes = self.companies['company'][company]['text'][start:index_com]
            label_nodes = self.companies['company'][company]['label'][start:index_com]

            text_len = self.companies['company'][company]['text_len'][start:index_com]
            label_len = self.companies['company'][company]['label_len'][start:index_com]
        assert len(nodes) == len(text_nodes)
        g, edges_weight = self.graphs[len(nodes) - 1], self.weights[len(nodes) - 1]

        label_feats = label_nodes
        text_feats = text_nodes

        # text_feats = pad_sequence(text_feats, batch_first=True, padding_value=0)
        # print(text_feats.shape, text_feats)

        item_dt.append(company)
        item_dt.append((g, edges_weight, label_feats, text_feats, text_len, label_len, not_initial))

        for i, key in enumerate(self.keys):
            # print("key is:{}, item:{}".format(key, self.data[key][index]))
            if i > 0:
                item_dt.append(self._create_onehot_labels(self.data[key][index], self.total_num[i-1]))
            else:
                item_dt.append(self.data[key][index])

        return item_dt

    def __len__(self):
        return self.length

    def get_edges_weight(self, patents, max_window=10):
        lengs = len(patents)
        edges_weight_dict = torch.zeros(lengs, lengs)
        for i in range(lengs):
            edges_weight_dict[i, i] += 1.
            for j in range(i + 1, min(lengs, i + 1 + max_window)):
                edges_weight_dict[i, j] += 1. / (j - i + 1)
                # edges_weight_dict[j, i] += 1. / (j - i)
        return edges_weight_dict


def batch_iter(data: [torch.Tensor], company_dts, num_classes_list, batch_size, slide_length, window, type='train', shuffle=False):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: zip(data['pad_seqs'], data['section'], data['subsection'], data['group'],
               data['subgroup'], data['onehot_labels'], data['labels'])
        batch_size: The size of the data batch.
        num_epochs: The number of epochs.
        shuffle: Shuffle or not (default: True).
    Returns:
        A batch iterator for data set.
    """
    data_set = My_dataset(data, company_dts, num_classes_list, slide_length=slide_length, window=window, type=type)
    data_loader = DataLoader(data_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return data_loader


def load_adjacent(data):
    dt = np.load(data).astype('float32')
    torch_dt = torch.from_numpy(dt)
    print(torch_dt)
    return torch_dt


def load_hierarchy_tree(raw_path, data_type, hierarchy_names) -> [torch.Tensor, []]:
    hierarchy_dt = []
    total_number_pre = []
    total_number_aft = [0]
    for hier_name in hierarchy_names:
        hier_path = raw_path.format(data_type, hier_name)
        hier_data = pd.read_csv(hier_path)
        data = hier_data.values
        total_number_pre.append(max(data[:, 0]) + 1)
        total_number_aft.append(max(data[:, 1]) + 1)
        hierarchy_dt.append(torch.tensor(data, dtype=torch.int))
        # print("hierarchy size for layer {}:{}".format(hier_name, hierarchy_dt[-1].shape))
    total_number_aft.append(0)
    # total_number_pre.append(max(data[:, 1]) + 1)

    total_num = []
    for low_count, high_count in zip(total_number_pre, total_number_aft):
        count = max(low_count, high_count)
        total_num.append(count)

    # print("classification codes in hierarchical tree is:{}".format(total_num))

    hierarchy_matrix = []
    pre_num = total_num[0]
    for aft_id in torch.arange(1, len(total_num)):
        aft_num = total_num[aft_id]
        matrix = torch.zeros(pre_num, aft_num)
        for line in hierarchy_dt[aft_id - 1]:
            matrix[line[0]][line[1]] = 1.0
        hierarchy_matrix.append(matrix)
        pre_num = aft_num
    return hierarchy_matrix, total_num


if __name__ == '__main__':
    print("main")
