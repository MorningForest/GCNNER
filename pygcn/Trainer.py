from pygcn.ner_model import GCNNerModel
from pygcn.utils import conlleval, pad_sequences, batch_yield, tag2label, read_dictionary, build_embedding_matrix, get_logger, read_corpus
from pygcn.conf import config, args, paths, spacener
import os
# from hyperopt import fmin, Trials, hp, STATUS_OK, tpe
from skopt.utils import use_named_args
from skopt import gp_minimize
import codecs

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

@use_named_args(dimensions=spacener)
def target(dropout, hidden_layer_size, clip, optimizer, batch, lr):
    root = os.getcwd()
    # vocab
    word2id = read_dictionary(os.path.join(root, paths['root_data'], args.train_data, 'word2id.pkl'))
    # embed weight
    embedd_weight = build_embedding_matrix(os.path.join(root, paths['embedd_path']), word2id)
    # train && dev data
    train_data = read_corpus(os.path.join(root, paths['train_data_path']))
    dev_data = read_corpus(os.path.join(root, paths['dev_data_path']))
    res = 'best param: dropout:{} hidden_size:{} clip:{} optimize:{} batch:{} lr:{}\n'.format(
        dropout, hidden_layer_size, clip,
        optimizer, batch, lr
    )
    write_to_txt(root, res)
    print(res)
    model = GCNNerModel(
        args.embedding_size, dropout, hidden_layer_size, len(tag2label), word2id,
        embedd_weight, args.update_embedding, optimizer, clip,
        os.path.join(root, paths['summary_path']), config, args.epoch, tag2label, batch,
        os.path.join(root, paths['model_path']),
        lr, get_logger(os.path.join(root, paths['log_path'])), os.path.join(root, paths['result_path']), False,
        'att',
    )
    model._build_graph()
    f1 = model.train(train_data, dev_data)
    return -f1

def write_to_txt(root ,item):
    with codecs.open(os.path.join(root, 'data/renmin/get_param.txt'), "a+", encoding="utf-8") as fp:
        fp.write(item)

def get_param(iters=100):
    root = os.getcwd()
    # vocab
    word2id = read_dictionary(os.path.join(root, paths['root_data'], args.train_data, 'word2id.pkl'))
    # embed weight
    search_res = gp_minimize(
        func=target,
        dimensions=spacener,
        acq_func='EI',
        n_calls=iters
    )
    res = 'best param: dropout:{} hidden_size:{} clip:{} optimize:{} batch:{} lr:{}\n'.format(
        search_res.x[0], search_res.x[1], search_res.x[2],
        search_res.x[3], search_res.x[4], search_res.x[5]
    )
    write_to_txt(root, res)
    # space4ner = {
    #     'embedding_size': 300,
    #     'dropout': Real(low=0, high=1, name='dropout'),
    #     'hidden_layer_size': Integer(low=200, high=300, name='hidden_layer_size'),
    #     'output_size': len(tag2label),
    #     'vocab': word2id,
    #     'embedding_weight': embedd_weight,
    #     'update_embedding_weight': True,
    #     'optimizer': args.optimizer,
    #     'clip_grad': Real(low=5, high=10, name='clip'),
    #     'summary_path': os.path.join(root, paths['summary_path']),
    #     'config': config,
    #     'epoch': 200,
    #     'tag2label': tag2label,
    #     'batch_size': Integer(low=64, high=256, name='batch'),
    #     'model_path': os.path.join(root, paths['model_path']),
    #     'lr': Real(low=0.00001, high=0.1, name='lr'),
    #     'logger': get_logger(os.path.join(root, paths['log_path'])),
    #     'result_path': os.path.join(root, paths['result_path']),
    #     'shuffle': False,
    #     'flag': 'att',
    # }

if __name__ == '__main__':
    # train()
    get_param()




