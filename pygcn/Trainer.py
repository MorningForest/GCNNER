from ner_model import GCNNerModel
from utils import conlleval, pad_sequences, batch_yield, tag2label, read_dictionary, build_embedding_matrix, get_logger, read_corpus
from conf import config, args, paths
import os

def train():
    #vocab
    word2id = read_dictionary(os.path.join(paths['root_data'], args.train_data, 'word2id.pkl'))
    #embed weight
    embedd_weight = build_embedding_matrix(paths['embedd_path'], word2id)
    #train && dev data
    train_data = read_corpus(paths['train_data_path'])
    dev_data = read_corpus(paths['dev_data_path'])
    model = GCNNerModel(
        args.embedding_size, args.dropout, args.hidden_dim, len(tag2label), word2id,
        embedd_weight, args.update_embedding, args.optimizer, args.clip,
        paths['summary_path'], config, args.epoch, tag2label, args.batch_size, paths['model_path'],
        args.lr, get_logger(paths['log_path']), paths['result_path'], 'att'
    )
    model._build_graph()
    model.train(train_data, dev_data)
if __name__ == '__main__':
    train()




