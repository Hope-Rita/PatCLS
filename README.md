# Adaptive Taxonomy Learning and Historical Patterns Modelling for Patent Classification
## Datasets
We provide the preprocessed datasets and NLP pretrained Model at [here](https://drive.google.com/drive/folders/1WzhGAmG2woPJIiMenmcpG6iuopExwoG1?usp=sharing), which should be put in the ./ folder.

## To train patent classification model:
* on USPTO-200K datasets: ``python train.py --US US``
* on CNPTD-200K datasets: ``python train.py --US CN``

## Performance

## Environments:
* [PyTorch 1.7.1](https://pytorch.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [numpy](https://github.com/numpy/numpy)
