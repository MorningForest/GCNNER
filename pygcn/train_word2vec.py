# encoding:utf-8
import argparse
from word2Vec import word2vec
import codecs


def main():
    data_path = r"../data/renmin/train_data"
    test_path = r"../data/renmin/test_data"
    fp = codecs.open(data_path, 'r', 'utf-8')
    ft = codecs.open(test_path, 'r', 'utf-8')
    train_data = [line.split()[0] for line in fp.readlines() if len(line.strip()) > 0]
    test_data = [line.split()[0] for line in ft.readlines() if len(line.strip()) > 0]
    data = train_data + test_data
    word2vec_model = word2vec.Word2Vec(size=args['size'],
                                       window=args['window'],
                                       min_count=3,
                                       tag=args['tag'],
                                       save_dir=r"../data/renmin"
                                       )
    word2vec_model.train_w2v(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-t',
                    '--tag',
                    required=True,
                    type=str,
                    help='Seed for initializing training.')
    ap.add_argument('-s',
                    '--size',
                    required=True,
                    default=300,
                    type=int,
                    help='Batch size for dataset iterators')
    ap.add_argument('-w',
                    '--window',
                    default=5,
                    type=int,
                    help='Batch size for dataset iterators')

    args = vars(ap.parse_args())
    main()

    '''
    python train_word2vec.py --tag=word --size=300
    python train_word2vec.py --tag=word --size=250
    python train_word2vec.py --tag=char --size=300
    python train_word2vec.py --tag=char --size=250

    '''