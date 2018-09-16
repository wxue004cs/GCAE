from copy import copy
import argparse
from collections import defaultdict, Counter
from lxml import etree
from nltk.tokenize.moses import MosesTokenizer
import simplejson as json
import codecs
import random


from mydatasets import SemEval, SemEval_TD

rest_train = {16: '../SemEval2016Task5/ABSA16_Restaurants_Train_SB1_v2.xml',
              15: '../SemEval2015/ABSA15_Restaurants_Train_Final.xml',
              14: '../SemEval2014Task4/Restaurants_Train_v2.xml'}
rest_test = {16: '../SemEval2016Task5/EN_REST_SB1_TEST.xml.gold',
             15: '../SemEval2015/ABSA15_Restaurants_Test.xml',
             14: '../SemEval2014Task4/Restaurants_Test_Gold.xml'}

laptop_train = {16: '../SemEval2016Task5/ABSA16_Laptops_Train_SB1_v2.xml',
                15: '../SemEval2015/ABSA15_Laptops_Train_Data.xml',
                14: '../SemEval2014Task4/Laptop_Train_v2.xml'}
laptop_test = {16: '../SemEval2016Task5/EN_LAPT_SB1_TEST_.xml.gold',
               15: '../SemEval2015/ABSA15_Laptops_Test.xml',
               14: '../SemEval2014Task4/Laptops_Test_Gold.xml'}
ds_train = {'r': rest_train, 'l': laptop_train}
ds_test = {'r': rest_test, 'l': laptop_test}
ds_yelp = '../data/yelp/review.json'


def filter_14(dataset):
    for example in dataset:
        example = copy(example)
        correct_aspect_sentiment = dict()
        for k, v in example['aspect_sentiment'].items():
            if v in ['positive', 'negative']:
                correct_aspect_sentiment[k] = v
        example['aspect_sentiment'] = correct_aspect_sentiment
        if len(example['aspect_sentiment']) > 0:
            yield example


def filter_by_aspect(dataset, aspect_filter, use_attribute=False):
    for example in dataset:
        example = copy(example)
        aspect_sentiment = defaultdict(list)
        for a, b in example["aspect_sentiment"]:
            if aspect_filter is not None and a not in aspect_filter:
                continue
            if not use_attribute:
                new_a = a[:a.find('#')] if '#' in a else a
            else:
                new_a = a.replace('#', ' ').replace('_', ' ')
            aspect_sentiment[new_a].append(b)
        correct_aspect_sentiment = dict()
        for a in aspect_sentiment:
            c = Counter(aspect_sentiment[a])
            for s in ['positive', 'negative', 'neutral']:
                if s not in c:
                    c[s] = 0
            if c['positive'] == c['negative']:
                correct_aspect_sentiment[a] = 'neutral'
            elif c['positive'] > c['negative']:
                correct_aspect_sentiment[a] = 'positive'
            else:
                correct_aspect_sentiment[a] = 'negative'
        example['aspect_sentiment'] = list(correct_aspect_sentiment.items())
        if len(example['aspect_sentiment']) > 0:
            yield example


def read_sentence1516(file_path):
    dataset = []
    with open(file_path, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for review_xml in root:
            sentences_xml = review_xml.find("sentences")
            for sentence_xml in sentences_xml:
                example = dict()
                example["sentence"] = sentence_xml.find('text').text.lower()
                opinions_xml = sentence_xml.find('Opinions')
                if opinions_xml is None:
                    continue
                example["aspect_sentiment"] = []
                for opinion_xml in opinions_xml:
                    example["aspect_sentiment"].append((opinion_xml.attrib["category"].lower(), opinion_xml.attrib["polarity"]))
                dataset.append(example)
    return dataset


def read_sentence14(file_path):
    dataset = []
    with open(file_path, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()
            example["sentence"] = sentence.find('text').text.lower()
            categories = sentence.find("aspectCategories")
            example["aspect_sentiment"] = []
            for c in categories:
                aspect = c.attrib['category'].lower()
                if aspect == 'anecdotes/miscellaneous':
                    aspect = 'misc'
                example["aspect_sentiment"].append((aspect, c.attrib['polarity']))
            dataset.append(example)
    return dataset


def read_sentence14_target(file_path, max_offset_len=83):
    tk = MosesTokenizer()
    with open(file_path, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()
            example["sentence"] = sentence.find('text').text.lower()

            # for RAN
            tokens = tk.tokenize(example['sentence'])

            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            example["aspect_sentiment"] = []
            example["left_right"] = []
            example['offset'] = []

            for c in terms:
                target = c.attrib['term'].lower()
                example["aspect_sentiment"].append((target, c.attrib['polarity']))

                # for td lstm
                left_index = int(c.attrib['from'])
                right_index = int(c.attrib['to'])
                example["left_right"].append((example['sentence'][:right_index],
                                              example['sentence'][left_index:],
                                              c.attrib['polarity']))

                # for RAN
                left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))
                right_word_offset = len(tk.tokenize(example['sentence'][right_index:]))
                token_index = list(range(len(tokens)))
                token_length = float(len(token_index))
                for i in range(len(tokens)):
                    if i < left_word_offset:
                        token_index[i] = 1 - (left_word_offset - token_index[i]) / token_length
                    elif i >= right_word_offset:
                        token_index[i] = 1 - (token_index[i] - (len(tokens) - right_word_offset) + 1) / token_length
                    else:
                        token_index[i] = 0
                token_index += [-1.] * (max_offset_len - len(tokens))
                example['offset'].append((token_index, target, c.attrib['polarity']))
            yield example


def read_sentence1516_target(file_path,  max_offset_len=83):
    tk = MosesTokenizer()

    with open(file_path, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for review_xml in root:
            sentences_xml = review_xml.find("sentences")
            for sentence_xml in sentences_xml:
                example = dict()
                example["sentence"] = sentence_xml.find('text').text.lower()

                # for RAN
                tokens = tk.tokenize(example['sentence'])

                opinions_xml = sentence_xml.find('Opinions')
                if opinions_xml is None:
                    continue
                example["aspect_sentiment"] = {}
                example['left_right'] = []
                example['offset'] = []

                for opinion_xml in opinions_xml:
                    target = opinion_xml.attrib["target"].lower()
                    if target == 'null':
                        continue
                    example["aspect_sentiment"][target] = opinion_xml.attrib["polarity"]

                    # for td lstm
                    left_index = int(opinion_xml.attrib['from'])
                    right_index = int(opinion_xml.attrib['to'])

                    example["left_right"].append((example['sentence'][:left_index],
                                                  example['sentence'][right_index:],
                                                  opinion_xml.attrib['polarity']))

                    # for RAN
                    left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))
                    right_word_offset = len(tk.tokenize(example['sentence'][right_index:]))

                    token_index = list(range(len(tokens)))
                    token_length = float(len(token_index))
                    for i in range(len(tokens)):
                        if i < left_word_offset:
                            token_index[i] = 1 - (left_word_offset - token_index[i]) / token_length
                        elif i >= len(tokens) - right_word_offset:
                            token_index[i] = 1 - (token_index[i] - (len(tokens) - right_word_offset) + 1) / token_length
                        else:
                            token_index[i] = 0
                    token_index += [-1.] * (max_offset_len - len(tokens))
                    example['offset'].append((token_index, target, opinion_xml.attrib['polarity']))

                if len(example["aspect_sentiment"]) == 0:
                    continue
                yield example


def get_semeval(years, aspects, rest_lap='r', use_attribute=False, dedup=False):
    semeval16_train = read_sentence1516(ds_train[rest_lap][16])
    semeval16_train = list(filter_by_aspect(semeval16_train, aspects, use_attribute))
    print("# SemEval 16 Train: {0}".format(len(semeval16_train)))

    semeval15_train = read_sentence1516(ds_train[rest_lap][15])
    semeval15_train = list(filter_by_aspect(semeval15_train, aspects, use_attribute))
    print("# SemEval 15 Train: {0}".format(len(semeval15_train)))

    if rest_lap == 'r':
        semeval14_train = read_sentence14(ds_train[rest_lap][14])
        if 14 in years and len(years) == 1:
            # exp on rest [14], keep positive, negative, conflict, neutral
            semeval14_train = list(semeval14_train)
        else:
            # exp on rest [14 + 16]. On the same aspect:
            # rest [14]: conflict + neutral -> neutral
            # rest [16]: #postive == #negative -> neutral
            semeval14_train = list(filter_by_aspect(semeval14_train, aspects, use_attribute))
        print("# SemEval 14 Train: {0}".format(len(semeval14_train)))
    else:
        semeval14_train = []

    semeval_train = []
    sentences = []
    train_total = []
    if 14 in years:
        train_total += semeval14_train
    if 15 in years:
        train_total += semeval15_train
    if 16 in years:
        train_total += semeval16_train

    if dedup:
        dup = 0
        for e in train_total:
            s = e['sentence'].strip()
            e['sentence'] = s
            if s not in sentences:
                semeval_train.append(e)
                sentences.append(s)
            else:
                dup += 1
    else:
        dup = 0
        semeval_train = train_total
    print("# Train: {}\t# Dup: {}".format(len(semeval_train), dup))

    semeval16_test = read_sentence1516(ds_test[rest_lap][16])
    semeval16_test = list(filter_by_aspect(semeval16_test, aspects, use_attribute))
    print("# SemEval 16 Test: {0}".format(len(semeval16_test)))

    semeval15_test = read_sentence1516(ds_test[rest_lap][15])
    semeval15_test = list(filter_by_aspect(semeval15_test, aspects, use_attribute))
    print("# SemEval 15 Test: {0}".format(len(semeval15_test)))

    if rest_lap == 'r':
        semeval14_test = read_sentence14(ds_test[rest_lap][14])
        if 14 in years and len(years) == 1:
            # exp on rest [14], keep positive, negative, conflict, neutral
            semeval14_test = list(semeval14_test)
        else:
            semeval14_test = list(filter_by_aspect(semeval14_test, aspects, use_attribute))
            print("# SemEval 14 Test: {0}".format(len(semeval14_test)))
    else:
        semeval14_test = []

    semeval_test = []
    sentences = []
    test_total = []
    if 14 in years:
        test_total += semeval14_test
    if 15 in years:
        test_total += semeval15_test
    if 16 in years:
        test_total += semeval16_test

    dup = 0
    if dedup:
        for e in test_total:
            s = e['sentence'].strip()
            e['sentence'] = s
            if s not in sentences:
                semeval_test.append(e)
                sentences.append(s)
            else:
                dup += 1
    else:
        semeval_test = test_total
    print("# Test: {}\t # Dup: {}".format(len(semeval_test), dup))

    return semeval_train, semeval_test


def get_semeval_target(years, rest_lap='rest', dedup=False):
    # semeval16_train = list(read_sentence1516_target(ds_train[rest_lap][16]))
    # print("# SemEval 16 Train: {0}".format(len(semeval16_train)))
    #
    # semeval15_train = list(read_sentence1516(ds_train[rest_lap][15]))
    # print("# SemEval 15 Train: {0}".format(len(semeval15_train)))

    semeval14_train = list(read_sentence14_target(ds_train[rest_lap][14]))
    print("# SemEval 14 Train: {0}".format(len(semeval14_train)))

    semeval_train = []
    sentences = []
    train_total = []
    if 14 in years:
        train_total += semeval14_train
    # if 15 in years:
    #     train_total += semeval15_train
    # if 16 in years:
    #     train_total += semeval16_train
    def has_conflict(e):
        for k, v in e['aspect_sentiment'].items():
            if v == 'conflict':
                return True
        return False

    if dedup:
        for e in train_total:
            s = e['sentence'].strip()
            e['sentence'] = s
            if dedup and s not in sentences:
                semeval_train.append(e)
                sentences.append(s)
    else:
        semeval_train = train_total
    print("# Train: {0}".format(len(semeval_train)))

    # semeval16_test = list(read_sentence1516_target(ds_test[rest_lap][16]))
    # print("# SemEval 16 Test: {0}".format(len(semeval16_test)))
    #
    # semeval15_test = list(read_sentence1516_target("../data/SemEval2015/ABSA15_Restaurants_Test.xml"))
    # print("# SemEval 15 Test: {0}".format(len(semeval15_test)))

    semeval14_test = list(read_sentence14_target(ds_test[rest_lap][14]))
    print("# SemEval 14 Test: {0}".format(len(semeval14_test)))

    semeval_test = []
    sentences = []
    test_total = []
    if 14 in years:
        test_total += semeval14_test
    # if 15 in years:
    #     test_total += semeval15_test
    # if 16 in years:
    #     test_total += semeval16_test
    if dedup:
        for e in test_total:
            s = e['sentence'].strip()
            e['sentence'] = s
            if dedup and s not in sentences:
                semeval_test.append(e)
                sentences.append(s)
    else:
        semeval_test = test_total
    print("# Test: {0}".format(len(semeval_test)))
    return semeval_train, semeval_test


def print_unrolled_stats(unrolled_data):
    counter = dict()
    sentiment_counter = defaultdict(int)
    length_list = []
    tk = MosesTokenizer()

    aspects = set()
    for x in unrolled_data:
        aspects.add(x['aspect'])
    for a in aspects:
        counter[a] = defaultdict(int)
    for e in unrolled_data:
        counter[e['aspect']][e['sentiment']] += 1
        length_list.append(len(tk.tokenize((e['sentence']))))
    for aspect in sorted(counter.keys()):
        total = 0
        for sentiment in sorted(counter[aspect].keys()):
            print('# {}\t\t{}:\t{}'.format(aspect, sentiment, counter[aspect][sentiment]))
            total += counter[aspect][sentiment]
            sentiment_counter[sentiment] += counter[aspect][sentiment]
        counter[aspect]['total'] = total
        print('# {}\t\t{}:\t{}'.format(aspect, 'total', total))
        print()
    print(sentiment_counter)
    return counter


def print_unrolled_stats_atsa(unrolled_data):
    counter = defaultdict(int)
    length_list = []
    tk = MosesTokenizer()

    for e in unrolled_data:
        counter[e['sentiment']] += 1
        length_list.append(len(tk.tokenize((e['sentence']))))

    for sentiment in sorted(counter.keys()):
        print('#{}:\t{}'.format(sentiment, counter[sentiment]))

    return counter


def read_yelp(N):
    yelp_train = []
    yelp_test = []
    with codecs.open(ds_yelp, 'r', 'utf8') as fin:
        for line in fin:
            l = json.loads(line)
            example = dict()
            example["sentence"] = l['text']
            example['aspect_sentiment'] = dict()
            s = int(l['stars'])
            if s != 3:
                example['aspect_sentiment']['aspect'] = 'positive' if int(l['stars']) > 3 else 'negative'
                if len(l['text']) < 15:
                    continue
                if random.random() < 0.8:
                    yelp_train.append(example)
                else:
                    yelp_test.append(example)
            if len(yelp_train) > N:
                return yelp_train, yelp_test
    return yelp_train, yelp_test


if __name__ == '__main__':
    # statistics
    parser = argparse.ArgumentParser(description='SemEval Statistics')
    parser.add_argument('-years', type=str, default='14_15_16',
                        help='data sets specified by the year, use _ to concatenate')
    parser.add_argument('-task', type=str, default='acsa')
    parser.add_argument('-rest_lap', type=str, default='rest', help='restaurants or laptops')
    parser.add_argument('-use_attribute', action='store_true', default=False)

    args = parser.parse_args()
    aspects = ''
    years = [int(i) for i in args.years.split('_')]

    if args.task == "acsa":
    # ACSA
        train_data, test_data = get_semeval(years, None, args.rest_lap, args.use_attribute)
        print(len(train_data))
        unrolled_train, mixed_train = SemEval.unroll(train_data)
        unrolled_test, mixed_test = SemEval.unroll(test_data)

        with open("acsa_train.json", "w") as fopen:
            fopen.write(json.dumps(unrolled_train))
        with open("acsa_test.json", "w") as fopen:
            fopen.write(json.dumps(unrolled_test))
        with open('acsa_hard_train.json', 'w') as fopen:
            fopen.write(json.dumps(mixed_train))
        with open('acsa_hard_test.json', 'w') as fopen:
            fopen.write(json.dumps(mixed_test))


        print()
        print('# unique training sentences {}'.format(len(train_data)))
        print('# unique test sentences {}'.format(len(test_data)))
        print('# unrolled training sentences {}'.format(len(unrolled_train)))
        print('# unrolled test sentences {}'.format(len(unrolled_test)))
        stat_train = print_unrolled_stats_atsa(unrolled_train)
        print('--------------------------------------')
        stat_test = print_unrolled_stats_atsa(unrolled_test)
        print('--------------------------------------')
        print_unrolled_stats_atsa(mixed_train)
        print('--------------------------------------')
        print_unrolled_stats_atsa(mixed_test)
    else:
    # ATSA
        train_data, test_data = get_semeval_target([14], args.rest_lap)
        unrolled_train, mixed_train = SemEval.unroll(train_data)
        unrolled_test, mixed_test = SemEval.unroll(test_data)

        with open("atsa_train.json", "w") as fopen:
            fopen.write(json.dumps(unrolled_train))
        with open("atsa_test.json", "w") as fopen:
            fopen.write(json.dumps(unrolled_test))
        with open('atsa_hard_train.json', 'w') as fopen:
            fopen.write(json.dumps(mixed_train))
        with open('atsa_hard_test.json', 'w') as fopen:
            fopen.write(json.dumps(mixed_test))

        print()
        print('# unique training sentences {}'.format(len(train_data)))
        print('# unique test sentences {}'.format(len(test_data)))
        print('# unrolled training sentences {}'.format(len(unrolled_train)))
        print('# unrolled test sentences {}'.format(len(unrolled_test)))
        stat_train = print_unrolled_stats_atsa(unrolled_train)
        print('--------------------------------------')
        print_unrolled_stats_atsa(mixed_train)
        print('--------------------------------------')
        stat_test = print_unrolled_stats_atsa(unrolled_test)
        print('--------------------------------------')

        print_unrolled_stats_atsa(mixed_test)