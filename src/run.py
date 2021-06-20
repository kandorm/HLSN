import pickle
import numpy as np
import math, random, string, os, sys, io
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_l_bfgs_b, basinhopping
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from timeit import default_timer as timer
from argparse import ArgumentParser

from custom_metrics import hamming_score, f1
from custom_dataset import batch_maker, flattern_result
from DAMSL import DAMSL

from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope

# from transformers import BertTokenizer
# pretrained_weights = './bert/'
# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

tune_best_model = []

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def best_score_search(true_labels, predictions, f):
# https://discuss.pytorch.org/t/multilabel-classification-how-to-binarize-scores-how-to-learn-thresholds/25396
# https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/46abf834128f41f4e6d8040f474ec51973ea9332/src/p_metrics.py#L15-L53
    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        return - f(true_labels, pd.DataFrame(predictions).values > pd.DataFrame(threshold).values.reshape(1, len(predictions[0])))
        # return - f(np.array(true_labels), pd.DataFrame(predictions).values > pd.DataFrame(threshold).values)[2]

    # print(len(predictions[0]))
    # Initialization of best threshold search
    thr_0 = [0.20] * len(predictions[0])
    constraints = [(0.,1.)] * len(predictions[0])
    def bounds(**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= 1))
        tmin = bool(np.all(x >= 0)) 
        return tmax and tmin
    
    # Search using L-BFGS-B, the epsilon step must be big otherwise there is no gradient
    minimizer_kwargs = {"method": "L-BFGS-B",
                        "bounds":constraints,
                        "options":{
                            "eps": 0.05
                            }
                       }
    
    # We combine L-BFGS-B with Basinhopping for stochastic search with random steps
    print("===> Searching optimal threshold for each label")
    start_time = timer()
    
    opt_output = basinhopping(f_neg, thr_0,
                                stepsize = 0.1,
                                minimizer_kwargs=minimizer_kwargs,
                                niter=10,
                                accept_test=bounds)
    
    end_time = timer()
    print("===> Optimal threshold for each label:\n{}".format(opt_output.x))
    print("Threshold found in: %s seconds" % (end_time - start_time))
    
    score = - opt_output.fun
    return score, opt_output.x

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def ret_index (li, s):
    if s in li:
        return li.index(s)
    else:
        # print(s)
        return -1

def str2vector (li, str, text, max_utterance_lengths):
    if text:
        max_len = max_utterance_lengths 
        ret = [ li[s]+1 for s in str.split()]
        ret += [0] * (max_len - len(ret))

        # max_len = max_utterance_lengths
        # ret = tokenizer.encode(str, add_special_tokens=False)
        # if len(ret) > max_len-2:
        #     _l = ret[:(max_len-2)//2]
        #     _r = ret[(max_len-2)//2-(max_len-2):]
        #     ret = _l + _r
        # ret = [101] + ret + [102]  # 101[CLS]  102[SEP]  0[PAD]
        # ret += [0] * (max_len - len(ret))

    else:
        if type(str) is np.float64 or len(str) == 0:
            return [0] * len(li)
        count = [ ret_index(li, s) for s in str.split()]
        ret = [0] * len(li)
        for c in count:
            assert c >= 0
            ret[c] = 1
    return ret

def utt2mask(utterance):
    ret = []
    for u in utterance:
        if u in [0, 101, 102]:
            ret.append(0)
        else:
            ret.append(1)
    return ret

def ret_predict (predicts, thresholds, discount=1.0):
    thresholds = [t * discount for t in thresholds]
    ret = [int(val >= thresholds[idx]) for idx, val in enumerate(predicts)]
    if sum(ret) == 0:
        # weighted_predicts =[x/y for x, y in zip(predicts, thresholds)]
        # ret[max(enumerate(weighted_predicts),key=lambda x: x[1])[0]] = 1
        if discount < 1.0:
            ret = ret_predict(predicts, thresholds, discount)
        # ret[ms_tags.index('JK')] = 1
        # if not tuning:
        #     print(ret)
    return ret

def Find_Optimal_Cutoff (target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


def da_filter (label_dict, y):
    tmp_y = []
    for turn in y:
        tmp_turn = []
        for sent in turn:
            tmp_sent = sorted(sent.split())
            if len(tmp_sent) > 1 and ' '.join(tmp_sent) not in label_dict:
                label = random.choice(tmp_sent)
                tmp_turn.append(label)
                # print(sent, label)
            else:
                tmp_turn.append(sent)
        tmp_y.append(tmp_turn)
    return tmp_y

def vector2tags (l, ms_tags):
    assert len(l) == len(ms_tags)
    ret = ''
    for i, val in enumerate(l):
        if val == 1:
            ret = ret + ' ' + ms_tags[i]
    return ret

def qu_parser (path, max_length):
    all_dialogs = []
    all_tags = []
    new_dialogue = True
    vocabulary = set()
    dialog_utterances = None
    max_len = 0
    
    with open(path, 'r') as f:
        for line in f:
            if line == '\n':
                new_dialogue = True
                continue
            utterance = line.split('\t')
            words = utterance[1].split()[:-1][:max_length]
            tags = utterance[0].replace('_', ' ')
            if len(words) > max_len:
                max_len = len(words)
            vocabulary = vocabulary.union(words)
            if new_dialogue:
                if dialog_utterances is not None:
                    all_dialogs.append(dialog_utterances)
                    all_tags.append(dialog_tags)
                dialog_utterances = [' '.join(words)]
                dialog_tags = [tags]
                new_dialogue = False
            else:
                dialog_utterances.append(' '.join(words))
                dialog_tags.append(tags)

    return all_dialogs, all_tags, vocabulary, max_len

def main(args):

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    # data loading
    dim = args.dim
    seed = args.random
    max_length = args.max_len
    if args.pretrain:
        max_length = 200
    print()
    print("random seed", seed)
    print("word embedding dimension", dim)

    if sys.argv[1] == 'tune':
        tuning = True
    else:
        tuning = False

    # read in curpus
    ms_tags_P = ["Question", "Inform", "Casual", "Discussion"]
    ms_tags =  ["O", "Q", "I", "A", "D", "W", "E", "R", "S", "F", "H", "T", "J", "V", "M"]
    sw_tags = ['sd', 'b', 'sv', '%', 'aa', 'qy', 'ba', 'x', 'ny', 'fc', 'qw', 'nn', 'qy^d', 'bk', 'h', 'bf', '^q', 'bh', 'na', 'fo_o_fw_by_bc', 'ad', '^2', 'b^m', 'qo', 'qh', '^h', 'ar', 'ng', 'br', 'no', 'arp_nd', 'fp', 'qrr', 't3', 'oo_co_cc', 'aap_am', 't1', 'bd', 'qw^d', '^g', 'fa', 'ft', '+']

    da_map_id = [[] for _ in range(len(ms_tags_P))]
    for i, tg in enumerate(ms_tags):
        da_map_id[ms_tags_P.index(DAMSL.da_to_cf(tg))].append(i)

    if args.swda:
        ms_tags = sw_tags
    # nertypes = ['NUM', 'ORGANIZATION', 'PERSON', 'URL', 'LOCATION', 'EMAIL', 'TECH']
    nertypes = ['NUM', 'ORGANIZATION', 'PERSON', 'URL', 'LOCATION', 'TECH']
    # ms_entitiedbowpath = os.path.normpath("./data/msdialog/old/collapsed_msdialog.csv")
    if not args.qu:
        if args.pretrain:
            ms_entitiedbowpath = os.path.normpath("./data/source.csv")
        else:
            ms_entitiedbowpath = os.path.normpath("./data/mastodon.csv")

        df = pd.read_csv(ms_entitiedbowpath)

        # conversation_numbers = df['conversation_no']
        utterance_tags = df['tags']
        utterance_tags_P = df['tags_P']
        utterances = df['utterance']
        utterance_status = df['utterance_status']
        #utterance_lengths = df['utterance_lengths']

        #max_utterance_lengths = np.minimum(max(utterance_lengths), max_length)
        max_utterance_lengths = max_length
        print('max utterance length', max_utterance_lengths)


        all_dialogs = []
        all_tags = []
        all_tags_P = []

        for i in range(len(utterances)):
            if utterance_status[i] == "B":
                dialog_utterances = [' '.join(utterances[i].split()[:max_length])]
                dialog_tags = [utterance_tags[i]]
                dialog_tags_P = [utterance_tags_P[i]]

            else:
                dialog_utterances.append(' '.join(utterances[i].split()[:max_length]))
                dialog_tags.append(utterance_tags[i])
                dialog_tags_P.append(utterance_tags_P[i])
                if utterance_status[i] == 'E':
                    all_dialogs.append(dialog_utterances)
                    all_tags.append(dialog_tags)
                    all_tags_P.append(dialog_tags_P)

        """
        combo_dict = {}
        for turn in all_tags:
            for sent in turn:
                labels = ' '.join(sorted(sent.split()))
                combo_dict[labels] = combo_dict.setdefault(labels, 0) + 1

        sorted_combos = sorted(combo_dict.items(), key=lambda x: x[1], reverse=True)
        label_dict = {item[0]: item[1] for item in sorted_combos[:32]}

        dialog_lengths = [len(dialog) for dialog in all_dialogs]
        """
        # print(label_dict)

        if args.pretrain:
            X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(all_dialogs, all_tags, all_tags_P, test_size=0.1, random_state=seed)
            X_test, y_test, z_test = X_val, y_val, z_val
        else:
            #X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(all_dialogs, all_tags, all_tags_P, test_size=0.1, random_state=seed)
            #X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X_train, y_train, z_train, test_size=0.1, random_state=seed)
            
            X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(all_dialogs, all_tags, all_tags_P, test_size=267/535, shuffle=False)
            X_test, y_test, z_test = X_val, y_val, z_val

        """
        if args.da_filter:
            print("[uncommon DAs filted]")
            y_train = da_filter(label_dict, y_train)
            y_val = da_filter(label_dict, y_val)
            y_test = da_filter(label_dict, y_test)
        """
    else:
        train_data_path = os.path.normpath('./data/msdialog/qu/train.tsv')
        valid_data_path = os.path.normpath('./data/msdialog/qu/valid.tsv')
        test_data_path = os.path.normpath('./data/msdialog/qu/test.tsv')
        X_train, y_train, v_train, max_len_train = qu_parser(train_data_path, max_length)
        X_val, y_val, v_val, max_len_val = qu_parser(valid_data_path, max_length)
        X_test, y_test, v_test, max_len_test = qu_parser(test_data_path, max_length)

        max_utterance_lengths = max([max_len_train, max_len_val, max_len_test])
        print('max utterance length', max_utterance_lengths)

    counts_train = [len(x) for x in y_train]
    counts_test = [len(x) for x in y_test]
    counts_val = [len(x) for x in y_val]

    print('Statistics of training set:')
    print('Utterances:', sum(counts_train))
    print('Min. # Turns Per Dialog', min(counts_train))
    print('Max. # Turns Per Dialog', max(counts_train))
    print('Avg. # Turns Per Dialog:', sum(counts_train)/len(counts_train))
    print('Avg. # DAs Per Utterance', sum(sum(len(y.split()) for y in x) for x in y_train)/sum(counts_train))
    print('Avg. # Words Per Utterance', sum(sum(len(y.split()) for y in x) for x in X_train)/sum(counts_train))
    print()
    print('Statistics of validation set:')
    print('Utterances:', sum(counts_val))
    print('Min. # Turns Per Dialog', min(counts_val))
    print('Max. # Turns Per Dialog', max(counts_val))
    print('Avg. # Turns Per Dialog:', sum(counts_val)/len(counts_val))
    print('Avg. # DAs Per Utterance', sum(sum(len(y.split()) for y in x) for x in y_val)/sum(counts_val))
    print('Avg. # Words Per Utterance', sum(sum(len(y.split()) for y in x) for x in X_val)/sum(counts_val))
    print()
    print('Statistics testing sets:')
    print('Utterances:', sum(counts_test))
    print('Min. # Turns Per Dialog', min(counts_test))
    print('Max. # Turns Per Dialog', max(counts_test))
    print('Avg. # Turns Per Dialog:', sum(counts_test)/len(counts_test))
    print('Avg. # DAs Per Utterance', sum(sum(len(y.split()) for y in x) for x in y_test)/sum(counts_test))
    print('Avg. # Words Per Utterance', sum(sum(len(y.split()) for y in x) for x in X_test)/sum(counts_test))

    # read in dict
    if not args.qu:
        bow_dict = {}
        target_vocab = []
        with open("./data/bow.tab", 'r', encoding="utf-8") as f:
        # with open("./data/msdialog/old/entitied_bow.tab", 'r') as f:
            for line in f:
                items = line.split('\t')
                key, value = items[0], int(items[1])
                # bow_dict[key] = value
                
                # if int(value) > 5:
                target_vocab.append(key)

        # if args.remove_ne:
        #     print("[name entity tags removed]")
        #     for term in nertypes:
        #         del bow_dict[term]
    else:
        target_vocab = v_train.union(v_val).union(v_test)

    output_size = len(ms_tags)
    output_size_P = len(ms_tags_P)

    word_to_ix = {word: i for i, word in enumerate(target_vocab)}

    """
    # embedding begins!
    glove_path = './data/yue/embeddings/glove'
    if args.msdialog:
        print('[Using msdialog corpus]')
        vectors = bcolz.open(glove_path+'/msdialog_embeddings.'+str(dim)+'.dat')[:]
        words = pickle.load(open(glove_path+'/msdialog_embeddings.'+str(dim)+'_words.pkl', 'rb'))
        word2idx = pickle.load(open(glove_path+'/msdialog_embeddings.'+str(dim)+'_idx.pkl', 'rb'))
    else:
        vectors = bcolz.open(glove_path+'/6B.'+str(dim)+'.dat')[:]
        words = pickle.load(open(glove_path+'/6B.'+str(dim)+'_words.pkl', 'rb'))
        word2idx = pickle.load(open(glove_path+'/6B.'+str(dim)+'_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    """
    glove = {}
    with open("./data/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            itmes = line.strip().split()
            glove[itmes[0]] = list(map(float, itmes[1:]))
    # print(glove['The'])
    # print(glove['.com'])

    # with padding vector
    matrix_len = len(target_vocab) + 1
    weights_matrix = np.zeros((matrix_len, dim))

    # the padding vector
    weights_matrix[0] = np.zeros((dim, ))

    words_found = 0

    for i, word in enumerate(target_vocab):
        try: 
            weights_matrix[i+1] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i+1] = np.random.normal(scale=0.6, size=(dim, ))

    print(words_found,'/',len(target_vocab), 'words found embeddings')

    print('Preparing data ...')

    train_n_iters = len(X_train)

    train_data = [ [str2vector(word_to_ix, sent, True, max_utterance_lengths) for sent in X_train[i]] for i in range(train_n_iters)]
    train_mask = [ [utt2mask(sent) for sent in train_data[i]] for i in range(train_n_iters)]
    train_target = [ [str2vector(ms_tags, sent, False, max_utterance_lengths) for sent in y_train[i]] for i in range(train_n_iters)]
    train_target_P = [ [str2vector(ms_tags_P, sent, False, max_utterance_lengths) for sent in z_train[i]] for i in range(train_n_iters)]

    # train_loader = DAMICDataset(train_data, train_target)


    val_n_iters = len(X_val)

    val_data = [ [str2vector(word_to_ix, sent, True, max_utterance_lengths) for sent in X_val[i]] for i in range(val_n_iters)]
    val_mask = [ [utt2mask(sent) for sent in val_data[i]] for i in range(val_n_iters)]
    val_target = [ [str2vector(ms_tags, sent, False, max_utterance_lengths) for sent in y_val[i]] for i in range(val_n_iters)]
    val_target_P = [ [str2vector(ms_tags_P, sent, False, max_utterance_lengths) for sent in z_val[i]] for i in range(val_n_iters)]
    # val_loader = DAMICDataset(val_data, val_target)


    test_n_iters = len(X_test)

    test_data = [ [str2vector(word_to_ix, sent, True, max_utterance_lengths) for sent in X_test[i]] for i in range(test_n_iters)]
    test_mask = [ [utt2mask(sent) for sent in test_data[i]] for i in range(test_n_iters)]
    test_target = [ [str2vector(ms_tags, sent, False, max_utterance_lengths) for sent in y_test[i]] for i in range(test_n_iters)]
    test_target_P = [ [str2vector(ms_tags_P, sent, False, max_utterance_lengths) for sent in z_test[i]] for i in range(test_n_iters)]

    # test_loader = DAMICDataset(test_data, test_target)

    if not tuning:
        run(args, da_map_id, weights_matrix, output_size, output_size_P, train_data, train_mask, train_target, train_target_P, val_data, val_mask, val_target, val_target_P, test_data, test_mask, test_target, test_target_P, tuning, ms_tags)
    else:
        def objective(params):
            if 'lstm_layers' in params.keys():
                args.lstm_layers = params['lstm_layers']
            if 'lstm_hidden' in params.keys():
                args.lstm_hidden = params['lstm_hidden']
            # args.dim = params['dim']
            if 'lr' in params.keys():
                args.lr = params['lr']
            if 'filters' in params.keys():
                args.filters = params['filters']
            # args.filter_sizes = params['filter_sizes']
            if 'cd' in params.keys():
                args.cd = params['cd']
            if 'tf' in params.keys():
                args.tf = params['tf']
            if 'ld' in params.keys():
                args.ld = params['ld']
            if 'k' in params.keys():
                args.k = params['k']
            if 'nheads' in params.keys():
                args.nheads = params['nheads']
            if 'nstack' in params.keys():
                args.nstack = params['nstack']
            if 'ploss' in params.keys():
                args.ploss = params['ploss']
            # args.max_len = params['max_len']
            return run(args, da_map_id, weights_matrix, output_size, output_size_P, train_data, train_mask, train_target, train_target_P, val_data, val_mask, val_target, val_target_P, test_data, test_mask, test_target, test_target_P, tuning, ms_tags)

        # # DAMIC WD
        # space = {
        #     'lstm_layers': scope.int(hp.quniform('lstm_layers', 4, 8, 1)),
        #     'lstm_hidden': scope.int(hp.quniform('lstm_hidden', 300, 500, 20)),
        #     # 'dim': scope.int(hp.quniform('dim', 100, 300, 100)),
        #     'lr': hp.quniform('lr', 0.0008, 0.002, 0.0001),
        #     'filters': scope.int(hp.quniform('filters', 150, 250, 20)),
        #     # 'filter_sizes': scope.int(hp.quniform('filter_sizes', 3, 6, 1)),
        #     'cd': hp.quniform('cd', 0.4, 0.8, 0.1),
        #     'tf': hp.quniform('tf', 0.5, 0.9, 0.1),
        #     'ld': hp.quniform('ld', 0.1, 0.2, 0.05),
        #     # 'max_len': scope.int(hp.quniform('max_len', 800, 1200, 100)),
        #     # 'batch_size': scope.int(hp.quniform('batch_size', 10, 100, 10)),
        # }

        # DAMIC
        # space = {
        #     'lstm_layers': scope.int(hp.quniform('lstm_layers', 2, 6, 1)),
        #     'lstm_hidden': scope.int(hp.quniform('lstm_hidden', 200, 1500, 100)),
        #     # 'dim': scope.int(hp.quniform('dim', 100, 300, 100)),
        #     'lr': hp.quniform('lr', 0.0008, 0.002, 0.0001),
        #     'filters': scope.int(hp.quniform('filters', 100, 300, 50)),
        #     # 'filter_sizes': scope.int(hp.quniform('filter_sizes', 3, 6, 1)),
        #     'cd': hp.quniform('cd', 0.1, 0.4, 0.1),
        #     # 'tf': hp.quniform('tf', 0.2, 0.9, 0.1),
        #     'ld': hp.quniform('ld', 0.05, 0.2, 0.05),
        #     # 'max_len': scope.int(hp.quniform('max_len', 800, 1200, 100)),
        #     # 'batch_size': scope.int(hp.quniform('batch_size', 10, 100, 10)),
        #     'k': scope.int(hp.quniform('k', 1, 6, 1)),
        # }

        # DAMIC kmax
        space = {
            # 'lstm_layers': scope.int(hp.quniform('lstm_layers', 2, 6, 1)),
            'lstm_hidden': scope.int(hp.quniform('lstm_hidden', 400, 800, 200)),
            # 'dim': scope.int(hp.quniform('dim', 100, 300, 100)),
            'lr': hp.quniform('lr', 1e-3, 2e-3, 1e-4),
            #'filters': scope.int(hp.quniform('filters', 200, 600, 100)),
            # 'filter_sizes': scope.int(hp.quniform('filter_sizes', 3, 6, 1)),
            # 'cd': hp.quniform('cd', 0.1, 0.5, 0.1),
            # 'tf': hp.quniform('tf', 0.2, 0.9, 0.1),
            'ld': hp.quniform('ld', 0.1, 0.5, 0.1),
            # 'max_len': scope.int(hp.quniform('max_len', 800, 1200, 100)),
            # 'batch_size': scope.int(hp.quniform('batch_size', 10, 100, 10)),
            #'k': scope.int(hp.quniform('k', 1, 10, 1)),
            'nheads': scope.int(hp.choice('nheads', [4,5,8,10])),
            'nstack': scope.int(hp.quniform('nstack', 2, 3, 1)),
            'ploss': hp.quniform('ploss', 0.5, 1.0, 0.1),
        }

        # DAMIC Stacked
        # space = {
        #     'lstm_layers': scope.int(hp.quniform('lstm_layers', 2, 6, 1)),
        #     'lstm_hidden': scope.int(hp.quniform('lstm_hidden', 100, 500, 100)),
        #     # 'dim': scope.int(hp.quniform('dim', 100, 300, 100)),
        #     'lr': hp.quniform('lr', 0.0001, 0.002, 0.0001),
        #     # 'filters': scope.int(hp.quniform('filters', 100, 300, 50)),
        #     # 'filter_sizes': scope.int(hp.quniform('filter_sizes', 3, 6, 1)),
        #     # 'cd': hp.quniform('cd', 0.2, 0.6, 0.1),
        #     # 'tf': hp.quniform('tf', 0.2, 0.9, 0.1),
        #     'ld': hp.quniform('ld', 0.1, 0.3, 0.05),
        #     # 'max_len': scope.int(hp.quniform('max_len', 800, 1200, 100)),
        #     # 'batch_size': scope.int(hp.quniform('batch_size', 10, 100, 10)),
        # }

        # baseline
        # space = {
        #     # 'lstm_layers': scope.int(hp.quniform('lstm_layers', 2, 6, 1)),
        #     'lstm_hidden': scope.int(hp.quniform('lstm_hidden', 100, 1500, 100)),
        #     # 'dim': scope.int(hp.quniform('dim', 100, 300, 100)),
        #     'lr': hp.quniform('lr', 0.0008, 0.002, 0.0001),
        #     'filters': scope.int(hp.quniform('filters', 100, 300, 50)),
        #     # 'filter_sizes': scope.int(hp.quniform('filter_sizes', 3, 6, 1)),
        #     'cd': hp.quniform('cd', 0.4, 0.8, 0.1),
        #     # 'tf': hp.quniform('tf', 0.2, 0.9, 0.1),
        #     # 'ld': hp.quniform('ld', 0.1, 0.3, 0.05),
        #     # 'max_len': scope.int(hp.quniform('max_len', 800, 1200, 100)),
        #     # 'batch_size': scope.int(hp.quniform('batch_size', 10, 100, 10)),
        # }


        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

        best_params = space_eval(space,best_params)

        print(best_params)

        if args.baseline:
            model_name = "baseline"
        elif args.wd:
            model_name = "DAMIC_wd"
        elif args.stacked:
            model_name = "DAMIC_stacked"
        else:
            model_name = "DAMIC"

        """
        with io.open("./output/"+model_name+"_best_params.tab",'a',encoding="utf8") as bp:
            if not args.qu:
                corpus = args.data_file
            else:
                corpus = "qu"
            for k, v in best_params.items():
                bp.write("\t".join([corpus, model_name, k, str(v)])+"\n")
        """
        if not args.pretrain:
            tune_acc_best = sorted(tune_best_model, key=lambda x:x[0], reverse=True)
            tune_f1_best = sorted(tune_best_model, key=lambda x:x[3], reverse=True)
            with io.open("acc.log", 'w', encoding="utf-8") as f:
                for item in tune_acc_best:
                    f.write(str(item))
                    f.write('\n')
            with io.open("f1.log", 'w', encoding="utf-8") as f:
                for item in tune_f1_best:
                    f.write(str(item))
                    f.write('\n')
        return best_params


def evalDA(haty, goldy, nclasses=15):
    nok = [0.]*nclasses
    nrec = [0.]*nclasses
    ntot = [0.]*nclasses
    for i in range(len(haty)):
        recy = haty[i]
        gldy = goldy[i]
        nrec[recy]+=1
        ntot[gldy]+=1
        if recy==gldy:
            nok[gldy]+=1

    nsamps = sum(ntot)
    preval=[float(ntot[i])/float(nsamps) for i in range(nclasses)]
    prec=0.
    reca=0.
    for j in range(nclasses):
        tp = nok[j]
        pr,re = 0.,0.
        if nrec[j]>0:
            pr=float(tp)/float(nrec[j])
        if ntot[j]>0:
            re=float(tp)/float(ntot[j])
        prec += pr*preval[j]
        reca += re*preval[j]
    if prec+reca==0.: 
        f1=0.
    else: 
        f1 = 2.*prec*reca/(prec+reca)
    acc = sum(nok) / nsamps
    return acc, prec, reca, f1

def run(args, da_map_id, weights_matrix, output_size, output_size_P, train_data, train_mask, train_target, train_target_P, val_data, val_mask, val_target, val_target_P, test_data, test_mask, test_target, test_target_P, tuning, ms_tags):
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.autograd import Variable
    from torch import optim
    from torch.utils import data as data_utils

    if args.baseline:
        from baseline import baseline as DAMIC
    elif args.wd:
        assert hasattr(args, 'tf')
        from DAMIC_wd import DAMIC
    elif args.stacked:
        from DAMIC_stacked import DAMIC
    else:
        from DAMIC import DAMIC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_matrix = torch.Tensor(weights_matrix)

    if tuning or sys.argv[1] == 'train':

        # Global setup
        hidden_size = args.lstm_hidden
        num_layers = args.lstm_layers
        n_epochs = args.epoch
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MultiLabelSoftMarginLoss()
        patient = args.patient
        learning_rate = args.lr
        bi_lstm = args.bi
        n_filters = args.filters
        filter_sizes = args.filter_sizes
        c_dropout = args.cd
        l_dropout = args.ld
        batch_size = args.batch_size
        gru = args.gru
        highway = args.highway
        kmax = args.k
        n_heads = args.nheads
        stack_num = args.nstack
        p_loss_ratio = args.ploss
        if hasattr(args, 'tf') and args.tf is not None:
            teacher_forcing_ratio = args.tf
        else:
            teacher_forcing_ratio = None

        save_path = './model/'+randomword(10)+'/'
        while os.path.exists(save_path):
            save_path = './model/'+randomword(10)+'/'

        if not tuning: 
            print()
            print('Parameters')
            print('lstm_hidden_size', hidden_size)
            print('lstm_layers', num_layers)
            print('epochs', n_epochs)
            print('patient', patient)
            print('learning_rate', learning_rate)
            print('bi_lstm', bi_lstm)
            print('n_filters', n_filters)
            print('filter_sizes', filter_sizes)
            print('batch_size', batch_size)
            print('CNN dropout', c_dropout)
            print('LSTM dropout', l_dropout)
            print('Teacher Forcing rate', teacher_forcing_ratio)
            print('GRU', gru)
            print('RNN Highway', highway)
            print('k max pooling', kmax)
            print()
        print('model will be saved to', save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args_file = open(save_path+"args", "w", encoding="utf-8")
        args_file.write(str(args))
        args_file.close()

        # torch.backends.cudnn.enabled = False
        model = DAMIC(hidden_size, output_size, output_size_P, weights_matrix, n_heads, stack_num, c_dropout, l_dropout, da_map_id)
        model = model.to(device)
        if args.pretrain:
            frozen_layers = [model.h2o, model.layer1, model.l_rnn]
            for layer in frozen_layers:
                for k, v in layer.named_parameters():
                    v.requires_grad = False
        base_params = filter(lambda p: p.requires_grad, model.parameters())
        if args.models:
            pre_model = torch.load(args.models[0]+str(args.model_file))
            model_dict = model.state_dict()
            pre_embed = pre_model["embedding.weight"]
            if pre_embed.size() != weights_matrix.size():
                x, y = pre_embed.size()
                new_embed = weights_matrix[:, :]
                new_embed[:x, :y] = pre_embed
                pre_model["embedding.weight"] = new_embed
            def filt_pretrain(pre_models, flt):
                for f in flt:
                    if f in pre_models:
                        return False
                return True
            pre_model = {k:v for k,v in pre_model.items() if filt_pretrain(k, ["layer1", "l_rnn", "h2o"])}
            model_dict.update(pre_model)
            model.load_state_dict(model_dict)

            h2o_params = list(map(id, model.h2o.parameters()))
            layer1_params = list(map(id, model.layer1.parameters()))
            l_rnn = list(map(id, model.l_rnn.parameters()))
            base_params = filter(lambda p: id(p) not in h2o_params+layer1_params+l_rnn, base_params)
            base_params = [{'params': base_params, 'lr': learning_rate / 10},
                           {'params': model.layer1.parameters(), 'lr': learning_rate},
                           {'params': model.l_rnn.parameters(), 'lr': learning_rate},
                           {'params': model.h2o.parameters(), 'lr': learning_rate}]

        # bert_params = list(map(id, model.context_embed.parameters()))
        # base_params = filter(lambda p: id(p) not in bert_params, model.parameters())
        # base_params = filter(lambda p: p.requires_grad, base_params)
        # params = [
        #     {'params': base_params, 'lr': learning_rate},
        #     {'params': model.context_embed.parameters(), 'lr': learning_rate / 100}
        # ]
        optimizer = optim.Adam(base_params, lr=learning_rate, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        if torch.cuda.device_count() > 1:
            if not tuning:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        losses = np.zeros(n_epochs)
        vlosses = np.zeros(n_epochs)

        best_epoch = 0
        stop_counter = 0
        best_score = None

        train_loader_dataset = batch_maker(train_data, train_mask, train_target, train_target_P, batch_size)
        val_loader_dataset = batch_maker(test_data, test_mask, test_target, test_target_P, batch_size)
        # learning
        for epoch in range(n_epochs):
            ###################
            # train the model #
            ###################
            model.train() # prep model for training

            for tr_i, data in enumerate(train_loader_dataset):
                src_seqs, seq_mask, trg_seqs, trg_seqs_P = data
                # inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
                src_seqs, seq_mask, trg_seqs, trg_seqs_P = src_seqs.to(device), seq_mask.to(device), trg_seqs.to(device), trg_seqs_P.to(device)

                outputs, outputs_P, lab_loss, lstm_out = model(src_seqs, seq_mask, trg_seqs)

                # print(outputs)
                outputs = outputs.to(device)
                outputs_P = outputs_P.to(device)

                optimizer.zero_grad()
                if args.pretrain:
                    loss = criterion(outputs_P.view(-1, output_size_P), trg_seqs_P.argmax(2).view(-1))
                else:
                    loss = criterion(outputs.view(-1, output_size), trg_seqs.argmax(2).view(-1)) + p_loss_ratio*criterion(outputs_P.view(-1, output_size_P), trg_seqs_P.argmax(2).view(-1)) + lab_loss

                loss.backward()
                optimizer.step()
                # print(loss.item())
                losses[epoch] += loss.item()

            if not tuning:
                print('epoch', epoch+1, ' average train loss: ', losses[epoch] / len(train_loader_dataset))

            scheduler.step()
            torch.cuda.empty_cache()

            ######################    
            # validate the model #
            ######################
            model.eval() # prep model for evaluation
            references = None
            predicts = None
            references_P = None
            predicts_P = None
            for data in val_loader_dataset:
                src_seqs, seq_mask, trg_seqs, trg_seqs_P = data
                src_seqs, seq_mask, trg_seqs, trg_seqs_P = src_seqs.to(device), seq_mask.to(device), trg_seqs.to(device), trg_seqs_P.to(device)

                outputs, outputs_P, lab_loss, lstm_out = model(src_seqs, seq_mask, trg_seqs)

                # print(outputs)
                outputs = outputs.to(device)
                outputs_P = outputs_P.to(device)
                if args.pretrain:
                    vlosses[epoch] += criterion(outputs_P.view(-1, output_size_P), trg_seqs_P.argmax(2).view(-1)).item()
                else:
                    vlosses[epoch] += criterion(outputs.view(-1, output_size), trg_seqs.argmax(2).view(-1)).item() + p_loss_ratio*criterion(outputs_P.view(-1, output_size_P), trg_seqs_P.argmax(2).view(-1)).item() + lab_loss.item()

                reference = flattern_result(trg_seqs.cpu().numpy())
                predict = flattern_result(outputs.detach().cpu().numpy())
                reference_P = flattern_result(trg_seqs_P.cpu().numpy())
                predict_P = flattern_result(outputs_P.detach().cpu().numpy())

                if references is None or predicts is None:
                    references = reference
                    predicts = predict
                else:
                    # print(predicts, predict)
                    references = np.append(references, reference, axis=0)
                    predicts = np.append(predicts, predict, axis=0)

                if references_P is None or predicts_P is None:
                    references_P = reference_P
                    predicts_P = predict_P
                else:
                    # print(predicts, predict)
                    references_P = np.append(references_P, reference_P, axis=0)
                    predicts_P = np.append(predicts_P, predict_P, axis=0)

            if not tuning:
                print('epoch', epoch+1, ' average val loss: ', vlosses[epoch] / len(val_loader_dataset))

            if args.pretrain:
                eval_scores = evalDA(np.argmax(predicts_P, axis=1), np.argmax(references_P, axis=1), nclasses=output_size_P)
            else:
                eval_scores = evalDA(np.argmax(predicts, axis=1), np.argmax(references, axis=1), nclasses=output_size)

            if best_score is None or eval_scores[-1] > best_score:
                best_score = eval_scores[-1]
                best_epoch = epoch+1
                torch.save(model.state_dict(), save_path+str(best_epoch))
                stop_counter = 0
                if not tuning:
                    print('epoch', best_epoch, 'model updated')
            else:
                stop_counter += 1
            torch.cuda.empty_cache()

            if stop_counter >= patient:
                print("Early stopping")
                break

        if not tuning:
            print('Models saved to', save_path)
            print('Best epoch', str(best_epoch), ', with score', str(best_score / len(val_loader_dataset)))


    if tuning or (sys.argv[1] == 'test' and len(sys.argv) > 2 and sys.argv[1] != ''):

        criterion = nn.CrossEntropyLoss()
        test_discount = 1.0

        if tuning:
            directory = save_path
            epoch = best_epoch
            result_file = ''
            loss_file = ''
            if teacher_forcing_ratio is not None:
                teacher_forcing_ratio = .0
        else:
            directory = args.models[0]
            epoch = args.model_file
            result_file = args.output_result[0]
            loss_file = args.output_loss

            # Global setup
            hidden_size = args.lstm_hidden
            num_layers = args.lstm_layers
            bi_lstm = args.bi
            n_filters = args.filters
            filter_sizes = args.filter_sizes
            c_dropout = args.cd
            l_dropout = args.ld
            test_discount = args.discount
            batch_size = args.batch_size
            gru = args.gru
            highway = args.highway
            kmax = args.k
            n_heads = args.nheads
            stack_num = args.nstack
            p_loss_ratio = args.ploss
            if hasattr(args, 'tf') and args.tf is not None:
                teacher_forcing_ratio = .0
            else:
                teacher_forcing_ratio = None
            

        if not tuning:
            print('lstm_hidden_size', hidden_size)
            print('lstm_layers', num_layers)
            print('bi_lstm', bi_lstm)
            print('n_filters', n_filters)
            print('filter_sizes', filter_sizes)
            print('batch_size', batch_size)
            print('CNN dropout', c_dropout)
            print('LSTM dropout', l_dropout)
            print('test discount', test_discount)
            print('Teacher Forcing rate', teacher_forcing_ratio)
            print('GRU', gru)
            print('RNN Highway', highway)
            print('k max pooling', kmax)


        if result_file and result_file != '':
            outf = open(result_file, 'w')
            out = 'dialogue_id, utterance_id, dialogue_length, utterance_length, utterance, references, predictions, hamming_score, p, r, f1\n'
        if loss_file and loss_file != '':
            lfile = open(loss_file, 'w')
            lout = ''

        bloss = 9999999.99;
        breferences = []
        bpredicts = []
        breferences_P = []
        bpredicts_P = []
        bfile = ''

        model = DAMIC(hidden_size, output_size, output_size_P, weights_matrix, n_heads, stack_num, c_dropout, l_dropout, da_map_id)
        model = model.to(device)

        """
        if torch.cuda.device_count() > 1:
            if not tuning:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        """

        """
        for filename in os.listdir(directory):
            if '.' in filename: continue
            # print('Epoch', filename)
            if loss_file and loss_file != '':
                lout = lout + filename
            if epoch > 0 and filename != str(epoch):
                # print('skipped')
                continue

            model.load_state_dict(torch.load(directory+filename))
            model.eval()

            train_loader_dataset = batch_maker(train_data, train_target, train_target_P, batch_size)
            val_loader_dataset = batch_maker(test_data, test_target, test_target_P, batch_size)
            test_loader_dataset = batch_maker(test_data, test_target, test_target_P, batch_size)

            loss = 0.0 # For plotting
            for data in train_loader_dataset:
                src_seqs, trg_seqs, trg_seqs_P = data
                src_seqs, trg_seqs, trg_seqs_P = src_seqs.to(device), trg_seqs.to(device), trg_seqs_P.to(device)

                outputs, outputs_P = model(src_seqs, trg_seqs)

                # print(outputs)
                outputs = outputs.to(device)
                outputs_P = outputs_P.to(device)
                if args.pretrain:
                    loss += margin_loss(outputs_P, trg_seqs_P).item()
                else:
                    loss += margin_loss(outputs, trg_seqs).item() + p_loss_ratio*margin_loss(outputs_P, trg_seqs_P).item()

            tloss = loss / len(train_loader_dataset)
            if loss_file and loss_file != '':
                lout = lout + ',' + str(tloss)
            if not tuning:
                print('Epoch', filename, 'average train loss: ', tloss)

            torch.cuda.empty_cache()

            loss = 0.0
            references = None
            predicts = None
            references_P = None
            predicts_P = None
            for data in val_loader_dataset:
                src_seqs, trg_seqs, trg_seqs_P = data
                src_seqs, trg_seqs, trg_seqs_P = src_seqs.to(device), trg_seqs.to(device), trg_seqs_P.to(device)

                outputs, outputs_P = model(src_seqs, trg_seqs)

                # print(outputs)
                outputs = outputs.to(device)
                outputs_P = outputs_P.to(device)
                if args.pretrain:
                    loss += margin_loss(outputs_P, trg_seqs_P).item()
                else:
                    loss += margin_loss(outputs, trg_seqs).item() + p_loss_ratio*margin_loss(outputs_P, trg_seqs_P).item()

                reference = flattern_result(trg_seqs.cpu().numpy())
                predict = flattern_result(outputs.detach().cpu().numpy())
                reference_P = flattern_result(trg_seqs_P.cpu().numpy())
                predict_P = flattern_result(outputs_P.detach().cpu().numpy())

                if references is None or predicts is None:
                    references = reference
                    predicts = predict
                else:
                    # print(predicts, predict)
                    references = np.append(references, reference, axis=0)
                    predicts = np.append(predicts, predict, axis=0)

                if references_P is None or predicts_P is None:
                    references_P = reference_P
                    predicts_P = predict_P
                else:
                    # print(predicts, predict)
                    references_P = np.append(references_P, reference_P, axis=0)
                    predicts_P = np.append(predicts_P, predict_P, axis=0)

            # print(references)

            vloss = loss / len(val_loader_dataset)
            if loss_file and loss_file != '':
                lout = lout + ',' + str(vloss) + '\n'
            if not tuning:
                print('Epoch', filename, 'average val loss: ', vloss)

            if vloss < bloss:
                bloss = vloss
                breferences = np.array(references)
                bpredicts = np.array(predicts)
                breferences_P = np.array(references_P)
                bpredicts_P = np.array(predicts_P)
                bfile = filename

            torch.cuda.empty_cache()

        if args.pretrain:
            best_score_P, thresholds_P = best_score_search(breferences_P, bpredicts_P, hamming_score)
        else:
            best_score, thresholds = best_score_search(breferences, bpredicts, hamming_score)
            best_score_P, thresholds_P = best_score_search(breferences_P, bpredicts_P, hamming_score)

        if not tuning:
            print('best validation epoch:', bfile, 'with score:', str(best_score))
        """

        test_loader_dataset = batch_maker(test_data, test_mask, test_target, test_target_P, batch_size)
        # load the best model
        model.load_state_dict(torch.load(directory+str(epoch)))
        model.eval()

        loss = 0.0 # For plotting
        references = None
        predicts = None
        references_P = None
        predicts_P = None
        utt_embed = None

        for data in test_loader_dataset:
            src_seqs, seq_mask, trg_seqs, trg_seqs_P = data
            src_seqs, seq_mask, trg_seqs, trg_seqs_P = src_seqs.to(device), seq_mask.to(device), trg_seqs.to(device), trg_seqs_P.to(device)

            outputs, outputs_P, lab_loss, lstm_out = model(src_seqs, seq_mask, trg_seqs)

            # print(outputs)
            outputs = outputs.to(device)
            outputs_P = outputs_P.to(device)
            if args.pretrain:
                loss += criterion(outputs_P.view(-1, output_size_P), trg_seqs_P.argmax(2).view(-1)).item()
            else:
                loss += criterion(outputs.view(-1, output_size), trg_seqs.argmax(2).view(-1)).item() + p_loss_ratio*criterion(outputs_P.view(-1, output_size_P), trg_seqs_P.argmax(2).view(-1)).item() + lab_loss.item()

            reference = flattern_result(trg_seqs.cpu().numpy())
            predict = flattern_result(outputs.detach().cpu().numpy())
            reference_P = flattern_result(trg_seqs_P.cpu().numpy())
            predict_P = flattern_result(outputs_P.detach().cpu().numpy())

            if sys.argv[1] == 'test':
                lstm_out = flattern_result(lstm_out.detach().cpu().numpy())
                if utt_embed is None:
                    utt_embed = lstm_out
                else:
                    utt_embed = np.append(utt_embed, lstm_out, axis=0)

            if references is None or predicts is None:
                references = reference
                predicts = predict
            else:
                references = np.append(references, reference, axis=0)
                predicts = np.append(predicts, predict, axis=0)

            if references_P is None or predicts_P is None:
                references_P = reference_P
                predicts_P = predict_P
            else:
                references_P = np.append(references_P, reference_P, axis=0)
                predicts_P = np.append(predicts_P, predict_P, axis=0)

            """
            if result_file and result_file != '':
                for d in src_seqs:
                    out = out + str(len(predict)) + ',' + str(len(X_test[i][j].split())) + ',"' + X_test[i][j] + '",' + vector2tags(r, ms_tags) + ',' + vector2tags(p, ms_tags) + ',' + str(hamming_score(r, p)) + ',' + str(f1(r, p)[0]) + ',' + str(f1(r, p)[1]) + ',' + str(f1(r, p)[2]) + '\n'
            """
        tloss = loss / len(test_loader_dataset)
        if not tuning:
            print('average test loss: ', tloss)

        torch.cuda.empty_cache()

        """
        predictions = []
        predictions_P = []
        if not args.pretrain:
            for j in range(len(predicts)):
                predictions.append(ret_predict(predicts[j], thresholds))
        for j in range(len(predicts_P)):
            predictions_P.append(ret_predict(predicts_P[j], thresholds_P))
        """
        # print(predictions)

        """
        references = np.array(references)
        predictions = np.array(predictions)
        references_P = np.array(references_P)
        predictions_P = np.array(predictions_P)
        """

        if sys.argv[1] == 'test':
            pre_tag = np.argmax(predicts, axis=1)
            ref_tag = np.argmax(references, axis=1)
            pre_P = np.argmax(predicts_P, axis=1)
            ref_P = np.argmax(references_P, axis=1)
            pickle.dump([utt_embed, ref_tag, pre_tag, ref_P, pre_P], open("utt_embed.dat", "wb"))

        if not args.pretrain:
            eval_scores = evalDA(np.argmax(predicts, axis=1), np.argmax(references, axis=1), nclasses=output_size)
            scores = ','.join([str(x) for x in eval_scores])
            print('Test Accuracy, Precision, Recall and F1 score: ', scores)

        eval_scores_P = evalDA(np.argmax(predicts_P, axis=1), np.argmax(references_P, axis=1), nclasses=output_size_P)
        scores = ','.join([str(x) for x in eval_scores_P])
        print('Test Accuracy, Precision, Recall and F1 score: ', scores)


        # f1 = f1_score(y_true=references, y_pred=predicts, average='weighted')
        # print('weighted F1 score: ', f1)
        if not args.pretrain and tuning:
            tun_model = [k for k in eval_scores]
            tun_model.append(directory)
            tune_best_model.append(tun_model)

        # print('weighted F1 score by chance: ', f1_score(y_true=references, y_pred=predicts_r, average='weighted'))
        """
        if not tuning:
            print('Tag',':','Accuracy, (Precision, Recall, F1)')
            for i in range(predictions.shape[1]):
                predictions_t = np.array([[p[i]] for p in predictions])
                references_t = np.array([[r[i]]for r in references])
                print(ms_tags[i], ':',hamming_score(y_true=references_t, y_pred=predictions_t),',', f1(y_true=references_t, y_pred=predictions_t))

            if result_file and result_file != '':
                outf.write(out)

            if loss_file and loss_file != '':
                lfile.write(lout)
        """
        if args.pretrain:
            return {'loss': -eval_scores_P[-1], 'status': STATUS_OK }
        else:
            return {'loss': -eval_scores[-1], 'status': STATUS_OK }

if __name__ == "__main__":
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(help='commands')

    # A train command
    train_parser = subparsers.add_parser('train', help='train the model')
    train_parser.add_argument('--models', type=str, nargs=1, help='directory for model files')
    train_parser.add_argument('--model_file', type=int, default=0, nargs='?', help='specify the epoch to test')
    train_parser.add_argument('--lstm_layers', type=int, default=2, nargs='?', help='number of layers in LSTM')
    train_parser.add_argument('--lstm_hidden', type=int, default=400, nargs='?', help='hidden size in output MLP')
    train_parser.add_argument('--dim', type=int, default=100, nargs='?', help='dimension of word embeddings')
    train_parser.add_argument('--epoch', type=int, default=1000, nargs='?', help='number of epochs to run')
    train_parser.add_argument("--remove_ne", type=str2bool, nargs='?',const=True, default=False, help="remove name entity tags")
    train_parser.add_argument("--da_filter", type=str2bool, nargs='?',const=True, default=False, help="filt uncommon DAs")
    train_parser.add_argument('--data_file', type=str, nargs='?', help='data file')
    train_parser.add_argument("--qu", type=str2bool, nargs='?',const=True, default=False, help="load data from Qu's dataset")
    train_parser.add_argument("--highway", type=str2bool, nargs='?',const=True, default=False, help="RNN input highway")
    train_parser.add_argument("--gru", type=str2bool, nargs='?',const=True, default=False, help="use GRU instead of LSTM")
    train_parser.add_argument('--patient', type=int, default=5, nargs='?', help='number of epochs to wait if no improvement and then stop the training.')
    train_parser.add_argument('--lr', type=float, default=0.001, nargs='?', help='learning rate')
    train_parser.add_argument("--bi", type=str2bool, nargs='?',const=True, default=True, help="Bi-LSTM")
    train_parser.add_argument('--filter_sizes', type=int, default=[3,4,5], nargs='+', help='filter sizes')
    train_parser.add_argument('--filters', type=int, default=200, nargs='?', help='number of CNN kernel filters.')
    train_parser.add_argument('--random', type=int, default=42, nargs='?', help='random seed')
    train_parser.add_argument('--cd', type=float, default=0.5, nargs='?', help='CNN dropout')
    train_parser.add_argument('--ld', type=float, default=0.05, nargs='?', help='LSTM dropout')
    train_parser.add_argument('--tf', type=float, default=None, nargs='?', help='Teacher Forcing rate')
    train_parser.add_argument('--max_len', type=int, default=70, nargs='?', help='max length of utterance')
    train_parser.add_argument("--msdialog", type=str2bool, nargs='?',const=True, default=False, help="msdialog embedding")
    train_parser.add_argument("--swda", type=str2bool, nargs='?',const=True, default=False, help="swda corpus")
    train_parser.add_argument('--batch_size', type=int, default=12, nargs='?', help='batch size')
    train_parser.add_argument('--gpu', type=int, default=[3,2,1,0], nargs='+', help='used gpu')
    train_parser.add_argument("--wd", type=str2bool, nargs='?',const=True, default=False, help="wide and deep mode")
    train_parser.add_argument("--baseline", type=str2bool, nargs='?',const=True, default=False, help="baseline mode")
    train_parser.add_argument("--stacked", type=str2bool, nargs='?',const=True, default=False, help="stacked RNN mode")
    train_parser.add_argument('--k', type=int, default=1, nargs='?', help='k max pooling')
    train_parser.add_argument('--nheads', type=int, default=8, nargs='?', help='multi-head attention hear numbers')
    train_parser.add_argument('--nstack', type=int, default=2, nargs='?', help='lastm-lan stack number')
    train_parser.add_argument("--pretrain", type=str2bool, nargs='?',const=True, default=False, help="multi corpus")
    train_parser.add_argument('--ploss', type=float, default=1.0, nargs='?', help='parent loss ratio')

    # A tuning command
    tune_parser = subparsers.add_parser('tune', help='tune the model')
    tune_parser.add_argument('--models', type=str, nargs=1, help='directory for model files')
    tune_parser.add_argument('--model_file', type=int, default=0, nargs='?', help='specify the epoch to test')
    tune_parser.add_argument('--lstm_layers', type=int, default=2, nargs='?', help='number of layers in LSTM')
    tune_parser.add_argument('--lstm_hidden', type=int, default=400, nargs='?', help='hidden size in output MLP')
    tune_parser.add_argument('--dim', type=int, default=100, nargs='?', help='dimension of word embeddings')
    tune_parser.add_argument('--epoch', type=int, default=1000, nargs='?', help='number of epochs to run')
    tune_parser.add_argument("--remove_ne", type=str2bool, nargs='?',const=True, default=False, help="remove name entity tags")
    tune_parser.add_argument("--da_filter", type=str2bool, nargs='?',const=True, default=False, help="filt uncommon DAs")
    tune_parser.add_argument('--data_file', type=str, nargs='?', help='data file')
    tune_parser.add_argument("--qu", type=str2bool, nargs='?',const=True, default=False, help="load data from Qu's dataset")
    tune_parser.add_argument("--highway", type=str2bool, nargs='?',const=True, default=False, help="RNN input highway")
    tune_parser.add_argument("--gru", type=str2bool, nargs='?',const=True, default=False, help="use GRU instead of LSTM")
    tune_parser.add_argument('--patient', type=int, default=5, nargs='?', help='number of epochs to wait if no improvement and then stop the training.')
    tune_parser.add_argument("--bi", type=str2bool, nargs='?',const=True, default=True, help="Bi-LSTM")
    tune_parser.add_argument('--filter_sizes', type=int, default=[3,4,5], nargs='+', help='filter sizes')
    tune_parser.add_argument('--filters', type=int, default=200, nargs='?', help='number of CNN kernel filters.')
    tune_parser.add_argument('--random', type=int, default=42, nargs='?', help='random seed')
    tune_parser.add_argument('--cd', type=float, default=0.5, nargs='?', help='CNN dropout')
    tune_parser.add_argument('--ld', type=float, default=0.05, nargs='?', help='LSTM dropout')
    tune_parser.add_argument('--max_len', type=int, default=70, nargs='?', help='max length of utterance')
    tune_parser.add_argument("--msdialog", type=str2bool, nargs='?',const=True, default=False, help="msdialog embedding")
    tune_parser.add_argument("--swda", type=str2bool, nargs='?',const=True, default=False, help="swda corpus")
    tune_parser.add_argument('--batch_size', type=int, default=12, nargs='?', help='batch size')
    tune_parser.add_argument('--gpu', type=int, default=[3,2,1,0], nargs='+', help='used gpu')
    tune_parser.add_argument("--wd", type=str2bool, nargs='?',const=True, default=False, help="wide and deep mode")
    tune_parser.add_argument("--baseline", type=str2bool, nargs='?',const=True, default=False, help="baseline mode")
    tune_parser.add_argument("--stacked", type=str2bool, nargs='?',const=True, default=False, help="stacked RNN mode")
    tune_parser.add_argument('--k', type=int, default=1, nargs='?', help='k max pooling')
    tune_parser.add_argument('--nheads', type=int, default=8, nargs='?', help='multi-head attention hear numbers')
    tune_parser.add_argument('--nstack', type=int, default=2, nargs='?', help='lastm-lan stack number')
    tune_parser.add_argument('--tf', type=float, default=None, nargs='?', help='Teacher Forcing rate')
    tune_parser.add_argument("--pretrain", type=str2bool, nargs='?',const=True, default=False, help="multi corpus")
    tune_parser.add_argument('--ploss', type=float, default=1.0, nargs='?', help='parent loss ratio')

     # A test command
    test_parser = subparsers.add_parser('test', help='test the model')
    test_parser.add_argument('--models', type=str, nargs=1, help='directory for model files', required=True)
    test_parser.add_argument('--model_file', type=int, default=0, nargs='?', help='specify the epoch to test')
    test_parser.add_argument('--lstm_layers', type=int, default=2, nargs='?', help='number of layers in LSTM')
    test_parser.add_argument('--lstm_hidden', type=int, default=400, nargs='?', help='hidden size in output MLP')
    test_parser.add_argument('--dim', type=int, default=100, nargs='?', help='dimension of word embeddings')
    test_parser.add_argument('--output_result', type=str, default=[''], nargs=1, help='file to store the test case result')
    test_parser.add_argument("--remove_ne", type=str2bool, nargs='?',const=True, default=False, help="remove name entity tags")
    test_parser.add_argument("--da_filter", type=str2bool, nargs='?',const=True, default=False, help="filt uncommon DAs")
    test_parser.add_argument('--data_file', type=str, nargs='?', help='data file')
    test_parser.add_argument("--qu", type=str2bool, nargs='?',const=True, default=False, help="load data from Qu's dataset")
    test_parser.add_argument("--highway", type=str2bool, nargs='?',const=True, default=False, help="RNN input highway")
    test_parser.add_argument("--gru", type=str2bool, nargs='?',const=True, default=False, help="use GRU instead of LSTM")
    test_parser.add_argument('--output_loss', type=str, nargs='?', help='loss output file')
    test_parser.add_argument("--bi", type=str2bool, nargs='?',const=True, default=True, help="Bi-LSTM")
    test_parser.add_argument('--filters', type=int, default=200, nargs='?', help='number of CNN kernel filters.')
    test_parser.add_argument('--filter_sizes', type=int, default=[3,4,5], nargs='+', help='filter sizes')
    test_parser.add_argument('--random', type=int, default=42, nargs='?', help='random seed')
    test_parser.add_argument('--cd', type=float, default=0.5, nargs='?', help='CNN dropout')
    test_parser.add_argument('--ld', type=float, default=0.05, nargs='?', help='LSTM dropout')
    test_parser.add_argument('--tf', type=float, default=None, nargs='?', help='Teacher Forcing rate')
    test_parser.add_argument('--max_len', type=int, default=70, nargs='?', help='max length of utterance')
    test_parser.add_argument("--msdialog", type=str2bool, nargs='?',const=True, default=False, help="msdialog embedding")
    test_parser.add_argument('--discount', type=float, default=1, nargs='?', help='test discount')
    test_parser.add_argument("--swda", type=str2bool, nargs='?',const=True, default=False, help="swda corpus")
    test_parser.add_argument('--batch_size', type=int, default=12, nargs='?', help='batch size')
    test_parser.add_argument('--gpu', type=int, default=[3,2,1,0], nargs='+', help='used gpu')
    test_parser.add_argument("--wd", type=str2bool, nargs='?',const=True, default=False, help="wide and deep mode")
    test_parser.add_argument("--baseline", type=str2bool, nargs='?',const=True, default=False, help="baseline mode")
    test_parser.add_argument("--stacked", type=str2bool, nargs='?',const=True, default=False, help="stacked RNN mode")
    test_parser.add_argument('--k', type=int, default=1, nargs='?', help='k max pooling')
    test_parser.add_argument('--nheads', type=int, default=8, nargs='?', help='multi-head attention hear numbers')
    test_parser.add_argument('--nstack', type=int, default=2, nargs='?', help='lastm-lan stack number')
    test_parser.add_argument("--pretrain", type=str2bool, nargs='?',const=True, default=False, help="multi corpus")
    test_parser.add_argument('--ploss', type=float, default=1.0, nargs='?', help='parent loss ratio')

    dataset_parser = subparsers.add_parser('dataset', help='save the dataset files')

    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    args = parser.parse_args()
    main(args)
