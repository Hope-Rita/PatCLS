import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run word2vec with hierarchy")

    # Data Parameters
    parser.add_argument("--data", default="Train on USPTO-200K", help="Descriptions of this running.")
    parser.add_argument("--gpu", type=int, default=1, help='Idx for the gpu to use')
    parser.add_argument("--US", type=str, default='CN', help='dataset of US or CN')
    parser.add_argument("--train-file", nargs="?",
                        default="../Data/{}/train.csv",
                        help="Training data.")
    parser.add_argument("--validation-file", nargs="?",
                        default="../Data/{}/validate.csv",
                        help="Validation data.")
    parser.add_argument("--test-file", nargs="?",
                        default="../Data/{}/test.csv",
                        help="Testing data.")

    parser.add_argument("--hierarchy_tree", nargs="?",
                        default="../Data/{}/tree/cls{}.csv",
                        help="hierarchy_tree")
    parser.add_argument("--hierarchy_names", type=list,
                        default=["I-II", "II-III", "III-IV"],
                        help="tree_names")
    parser.add_argument("--word2vec-file", nargs="?", default="../NLPModel/word2vec-{}.kv",
                        help="Word2vec file for embedding characters (the dim need to be the same as embedding dim).")

    # Model Hyperparameters
    parser.add_argument("--embedding-dim", type=int, default=100, help="Dimensionality of character embedding.")
    parser.add_argument("--lstm-dim", type=int, default=256, help="Dimensionality of LSTM neurons.")
    parser.add_argument("--fc-dim", type=int, default=256, help="Dimensionality for FC neurons.")
    parser.add_argument("--gcn_layer", type=int, default=2, help="Numbers for gcn layers")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout keep probability.")
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--fuse_layer', type=int, default=1, help='fuse embedding from 1-3 layers')

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--slide_length", type=int, default=50, help="total length for building a graph")
    parser.add_argument("--window", type=int, default=10, help="slide window for connections")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--decay-rate", type=float, default=0.95, help="Rate of decay for learning rate.")
    parser.add_argument("--decay-steps", type=int, default=100, help="How many steps before decay learning rate.")

    return parser.parse_args()
