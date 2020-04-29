from pygcn.ner_model import GCNNerModel
from pygcn.utils import conlleval, pad_sequences, batch_yield, tag2label, read_dictionary, build_embedding_matrix, get_logger, read_corpus
from pygcn.conf import config, args, paths
import os
from hyperopt import fmin, Trials, hp, STATUS_OK, tpe

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
        args.lr, get_logger(os.path.join(root, paths['log_path'])), os.path.join(root, paths['result_path']), False, 'att',
    )
    model._build_graph()
    model.train(train_data, dev_data)

def target(param):
    root = os.getcwd()
    # vocab
    word2id = read_dictionary(os.path.join(root, paths['root_data'], args.train_data, 'word2id.pkl'))
    # embed weight
    embedd_weight = build_embedding_matrix(os.path.join(root, paths['embedd_path']), word2id)
    # train && dev data
    train_data = read_corpus(os.path.join(root, paths['train_data_path']))
    dev_data = read_corpus(os.path.join(root, paths['dev_data_path']))
    model = GCNNerModel(param)
    model._build_graph()
    model.train(train_data, dev_data)

def get_param(iters):
    root = os.getcwd()
    # vocab
    word2id = read_dictionary(os.path.join(root, paths['root_data'], args.train_data, 'word2id.pkl'))
    # embed weight
    embedd_weight = build_embedding_matrix(os.path.join(root, paths['embedd_path']), word2id)

    space4ner = {
        'embedding_size': 300,
        'dropout': hp.uniform('dropout', 0, 1),
        'hidden_layer_size': hp.randint('hidden_layer_size', 200, 300),
        'output_size': len(tag2label),
        'vocab': word2id,
        'embedding_weight': embedd_weight,
        'update_embedding_weight': True,
        'optimizer': args.optimizer,
        'clip_grad': hp.uniform('clip', 5, 10),
        'summary_path': os.path.join(root, paths['summary_path']),
        'config': config,
        'epoch': 200,
        'tag2label': tag2label,
        'batch_size': hp.randint('batch', 64, 256),
        'model_path': os.path.join(root, paths['model_path']),
        'lr': hp.uniform('lr', 0.00001, 0.1),
        'logger': get_logger(os.path.join(root, paths['log_path'])),
        'result_path': os.path.join(root, paths['result_path']),
        'shuffle': False,
        'flag': 'att',
    }


if __name__ == '__main__':
    train()




