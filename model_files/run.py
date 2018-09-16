import os
import argparse
import datetime
import torch
import torchtext.data as data
from w2v import *

from cnn_gate_aspect_model import CNN_Gate_Aspect_Text
from cnn_gate_aspect_model_atsa import CNN_Gate_Aspect_Text as CNN_Gate_Aspect_Text_atsa



import mydatasets as mydatasets
from getsemeval import get_semeval, get_semeval_target, read_yelp
import cnn_train

parser = argparse.ArgumentParser(description='CNN text classificer')

# learning
parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate [default: 0.001]')
parser.add_argument('-l2', type=float, default=0, help='initial learning rate [default: 0]')
parser.add_argument('-momentum', type=float, default=0.99, help='SGD momentum [default: 0.99]')
parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 30]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('-grad_clip', type=float, default=5, help='max value of gradients')
parser.add_argument('-lr_decay', type=float, default=0, help='learning rate decay')

# logging
parser.add_argument('-log-interval',  type=int, default=10,   help='how many steps to wait before logging training status [default: 10]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=10000, help='how many steps to wait before saving [default:10000]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')

# data
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch [default: True]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-aspect_embed_dim', type=int, default=300, help='number of aspect embedding dimension [default: 300]')
parser.add_argument('-unif', type=float, help='Initializer bounds for embeddings', default=0.25)
parser.add_argument('-embed_file', default='w2v', help='w2v or glove')
parser.add_argument('-aspect_file', type=str, default='', help='aspect embedding')
parser.add_argument('-years', type=str, default='14_15_16', help='data sets specified by the year, use _ to concatenate')
parser.add_argument('-aspects', type=str, default='all', help='selected aspects, use _ to concatenate')
parser.add_argument('-atsa', action='store_true', default=False)
parser.add_argument('-r_l', type=str, default='r', help='restaurants or laptops')
parser.add_argument('-use_attribute', action='store_true', default=False)
parser.add_argument('-aspect_phrase', action='store_true', default=False)

# model CNNs
parser.add_argument('-model', type=str, default='CNN', help='Model name [default: CNN]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel [default: 100]')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-att_dsz', type=int, default=100, help='Attention dimension [default: 100]')
parser.add_argument('-att_method', type=str, default='concat', help='Attention method [default: concat]')

## CNN_CNN
parser.add_argument('-lambda_sm', type=float, default=1.0, help='Lambda weight for sentiment loss [default: 1.0]')
parser.add_argument('-lambda_as', type=float, default=1.0, help='Lambda weight for aspect loss [default: 1.0]')

## LSTM
parser.add_argument('-lstm_dsz', type=int, default=300, help='LSTM hidden layer dimension size [default: 300]')
parser.add_argument('-lstm_bidirectional', type=bool, default=True, help='is LSTM bidirecional [default: True]')
parser.add_argument('-lstm_nlayers', type=int, default=1, help='the number of layers of LSTM [default: 1]')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-sentence', type=str, default=None, help='predict the sentence given')
parser.add_argument('-target', type=str, default=None, help='predict the target given')

parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-verbose', type=int, default=0)
parser.add_argument('-trials', type=int, default=1, help='the number of trials')

args = parser.parse_args()

good_lap_attributes = ['battery#operation_performance', 'battery#quality', 'company#general', 'cpu#operation_performance', 'display#design_features', 'display#general', 'display#operation_performance', 'display#quality', 'display#usability', 'graphics#general', 'graphics#quality', 'hard_disc#design_features', 'hard_disc#quality', 'keyboard#design_features', 'keyboard#general', 'keyboard#operation_performance', 'keyboard#quality', 'keyboard#usability', 'laptop#connectivity', 'laptop#design_features', 'laptop#general', 'laptop#miscellaneous', 'laptop#operation_performance', 'laptop#portability', 'laptop#price', 'laptop#quality', 'laptop#usability', 'memory#design_features', 'motherboard#quality', 'mouse#design_features', 'mouse#general', 'mouse#operation_performance', 'mouse#quality', 'mouse#usability', 'multimedia_devices#general', 'multimedia_devices#operation_performance', 'multimedia_devices#quality', 'optical_drives#quality', 'os#general', 'os#operation_performance', 'os#usability', 'power_supply#quality', 'shipping#quality', 'software#design_features', 'software#general', 'software#miscellaneous', 'software#operation_performance', 'software#usability', 'support#price', 'support#quality']


def load_semeval_data(text_field, as_field, sm_field, years, aspects, **kargs):
    if not args.atsa:
        semeval_train, semeval_test = get_semeval(years, aspects, args.r_l, args.use_attribute)
    else:
        semeval_train, semeval_test = get_semeval_target(years, args.r_l)

    predict_test = [{"aspect": "food",
                     "sentiment": "positive",
                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"},
                    {"aspect": "service",
                     "sentiment": "negative",
                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"},
                    {"aspect": "service",
                     "sentiment": "negative",
                     "sentence": "good food in cute - though a bit dank - little hangout, but service terrible"}
                    ]
    predict_data = mydatasets.SemEval(text_field, as_field, sm_field, predict_test)

    train_data, dev_data, mixed_data = mydatasets.SemEval.splits_train_test(text_field, as_field, sm_field,
                                                     semeval_train, semeval_test)

    text_field.build_vocab(train_data, dev_data)
    as_field.build_vocab(train_data, dev_data)
    sm_field.build_vocab(train_data, dev_data)
    train_iter, test_iter, mixed_test_iter, predict_iter = data.Iterator.splits(
                                (train_data, dev_data, mixed_data, predict_data),
                                batch_sizes=(args.batch_size, len(dev_data), len(mixed_data), len(predict_data)),
                                **kargs)

    return train_iter, test_iter, mixed_test_iter, predict_iter


n_trials = args.trials
accuracy_trials = []
time_stamps_trials = []

# load data
print("Loading data...")
text_field = data.Field(lower=True, tokenize='moses')

if not args.aspect_phrase:
    as_field = data.Field(sequential=False)
else:
    print('phrase')
    as_field = data.Field(lower=True, tokenize='moses')

sm_field = data.Field(sequential=False)
years = [int(i) for i in args.years.split('_')]
aspects = None
if args.r_l == 'lap' and args.use_attribute:
    aspects = good_lap_attributes

train_iter, test_iter, mixed_test_iter, predict_iter = load_semeval_data(text_field, as_field, sm_field, years, aspects,
                                                          device=-1, repeat=False)

print('# aspects: {}'.format(len(as_field.vocab.stoi)))
print('# sentiments: {}'.format(len(sm_field.vocab.stoi)))

args.embed_num = len(text_field.vocab)
args.class_num = len(sm_field.vocab) - 1
args.aspect_num = len(as_field.vocab)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

for t in range(n_trials):
    if args.embed_file == 'w2v':
        print("Loading W2V pre-trained embedding...")
        word_vecs = load_w2v_embedding(text_field.vocab.itos, args.unif, 300)
    elif args.embed_file == 'glove':
        print("Loading GloVe pre-trained embedding...")
        word_vecs = load_glove_embedding(text_field.vocab.itos, args.unif, 300)
    else:
        raise(ValueError('Error embedding file'))
    print('# word initialized {}'.format(len(word_vecs)))

    print("Loading pre-trained aspect embedding...")
    if args.aspect_file == '':
        args.aspect_embedding = load_aspect_embedding_from_w2v(as_field.vocab.itos, text_field.vocab.stoi, word_vecs)
        args.aspect_embed_dim = args.embed_dim
    else:
        args.aspect_embedding, args.aspect_embed_dim = load_aspect_embedding_from_file(as_field.vocab.itos, args.aspect_file)

    args.embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
    args.aspect_embedding = torch.from_numpy(np.asarray(args.aspect_embedding, dtype=np.float32))

    print('# aspect embedding size: {}'.format(len(args.aspect_embedding)))
    # model

    if args.model == 'CNN_Basic':
        # vanilla CNN
        model = CNN_Basic(args)
        train = cnn_train
    elif args.model == 'LSTM':
        model = LSTM_Text(args)
        train = lstm_train
    elif args.model == 'CNN':
        # CAN
        model = CNN_Text(args)
        train = cnn_train
    elif args.model == 'CNN_CNN':
        # CAN-kv, CNN-kv-rl-e
        model = CNN2_Text(args)
        train = cnn2_train
    elif args.model == 'CNN_Deep':
        model = CNN_Deep_Text(args)
        train = cnn_train
    elif args.model == 'CNN_Gate':
        # GCAE no aspect embedding
        model = CNN_Gate_Text(args)
        train = cnn_train
    elif args.model == 'CNN_Gate_Aspect' and not args.atsa:
        # GCAE
        model = CNN_Gate_Aspect_Text(args)
        train = cnn_train
    elif args.model == 'CNN_Gate_Aspect' and args.atsa:
        # CNN on target expressions
        model = CNN_Gate_Aspect_Text_atsa(args)
        train = cnn_train
    elif args.model == 'CNN_Gate_Aspect_Average':
        model = CNN_Gate_Aspect_Text(args)
        train = cnn_train
    elif args.model == 'CNN_Gate_Att':
        # GCAE + Att
        model = CNN_Gate_ATT_Text(args)
        train = cnn_train
    elif args.model == 'CNN2_Gate_Att':
        # GCAE + Att +
        model = CNN2_Gate_ATT_Text(args)
        train = cnn2_train
    elif args.model == 'IAN':
        model = IAN(args)
        train = ian_train
    else:
        raise(ValueError('Error Model'))

    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        model.load_state_dict(torch.load(args.snapshot))

    model = model.cuda()

    # train or predict
    if args.sentence is not None:
        train = cnn_train
        label = train.predict(model, args.sentence, text_field, args.target, as_field, sm_field)
        print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
    elif args.test:
        try:
            train.eval(dev_iter, model, args)
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print()
        acc, time_stamps = train.train(train_iter, test_iter, mixed_test_iter, model, args, text_field, as_field, sm_field, predict_iter)
        accuracy_trials.append([acc[0], acc[1]])   # accuracy on test, accuracy on mixed
        time_stamps_trials.append(time_stamps)

print(accuracy_trials)
accuracy_trials = np.array(accuracy_trials)
means = accuracy_trials.mean(0)
stds = accuracy_trials.std(0)
print('{:.2f}    {:.2f}'.format(means[0], stds[0]))
print('{:.2f}    {:.2f}'.format(means[1], stds[1]))

with open('time_stamps', 'w') as fopen:
    for trials in time_stamps_trials:
        for acc, _ in trials:
            fopen.write('{:.4f} '.format(acc))
        fopen.write('\n')
        for _, dtime in trials:
            fopen.write('{:.4f} '.format(dtime))
        fopen.write('\n')
