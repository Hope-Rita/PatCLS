# Adaptive Taxonomy Learning and Historical Patterns Modelling for Patent Classification
## Datasets
We provide the preprocessed datasets and NLP pretrained Model at [here](https://drive.google.com/drive/folders/1WzhGAmG2woPJIiMenmcpG6iuopExwoG1?usp=sharing), which should be put in the ./ folder. The datasets contain two components: patent documents and a hierarchy of IPC classification codes. For patent documents, we provide anonymous IDs for companies, IPC codes in five levels, patent documents, publication times, and descriptions for patents. Under the "Tree" category, we store the hierarchical structure for IPC codes at adjacent levels. If you wish to utilize our work with your datasets, simply ensure they are formatted similarly and train a Word2Vec model as the pre-training model to capture textual information.

## To train patent classification model:
* on USPTO-200K datasets: ``python train.py --US US``
* on CNPTD-200K datasets: ``python train.py --US CN``

## Performance
| Datasets   | Precision(@1) | Recall(@1) | NDCG(@1) | Precision(@3) | Recall(@3) | NDCG(@3) | Precision(@5) | Recall(@5) | NDCG(@5) |
| ---------- | :-----------: | :--------: | :------: | :-----------: | :--------: | :------: | :-----------: | :--------: | :------: |
| USPTO-200K |    0.8341     |   0.5226   |  0.8341  |    0.4856     |   0.7794   |  0.8017  |    0.3342     |   0.8537   |  0.8255  |
| CNPTD-200K |    0.6848     |   0.5795   |  0.6848  |    0.3428     |   0.8084   |  0.7540  |    0.2257     |   0.8708   |  0.7814  |


## Environments:
* [PyTorch 1.7.1](https://pytorch.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [numpy](https://github.com/numpy/numpy)
