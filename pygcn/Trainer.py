from pygcn.ner_model import GCNNerModel
from pygcn.utils import conlleval, pad_sequences, batch_yield, tag2label, read_dictionary, build_embedding_matrix, get_logger, read_corpus
from pygcn.conf import config, args, paths
import os

def train():
    root = os.getcwd()
    #vocab
    word2id = read_dictionary(os.path.join(root, paths['root_data'], args.train_data, 'word2id.pkl'))
    #embed weight
    embedd_weight = build_embedding_matrix(os.path.join(root, paths['embedd_path']), word2id)
    #train && dev data
    train_data = read_corpus(os.path.join(root, paths['train_data_path']))
    dev_data = read_corpus(os.path.join(root, paths['dev_data_path']))
    model = GCNNerModel(
        args.embedding_size, args.dropout, args.hidden_dim, len(tag2label), word2id,
        embedd_weight, args.update_embedding, args.optimizer, args.clip,
        os.path.join(root, paths['summary_path']), config, args.epoch, tag2label, args.batch_size, os.path.join(root, paths['model_path']),
        args.lr, get_logger(os.path.join(root, paths['log_path'])), os.path.join(root, paths['result_path']), 'att'
    )
    model._build_graph()
    model.train(train_data, dev_data)
if __name__ == '__main__':
    train()




