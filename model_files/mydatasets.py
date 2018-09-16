import re
import os
import random
import tarfile
from six.moves import urllib
from torchtext import data
from torch.utils.data import Dataset


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = 'rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True ,root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


class SemEval(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, as_field, sm_field, input_data, **kwargs):
        """Create an SemEval dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            as_field: The field that will be used for aspect data.
            sm_field: The field that will be used for sentiment data.
            input_data: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('aspect', as_field), ('sentiment', sm_field)]

        examples = []
        for e in input_data:
            if 'pp.' in e['sentence']:
                continue
            examples.append(data.Example.fromlist([e['sentence'], e['aspect'], e['sentiment']], fields))
        super(SemEval, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def unroll(input_data):
        import json
        # return unrolled sentences and sentences which have multiple aspects and different sentiments
        unrolled = []
        mixed = []
        from collections import defaultdict
        total_counter = defaultdict(int)
        mixed_counter = defaultdict(int)

        for e in input_data:
            for aspect, sentiment in e['aspect_sentiment']:
                unrolled.append({'sentence': e['sentence'], 'aspect': aspect, 'sentiment': sentiment})
                if len(e['aspect_sentiment']) and len(set(map(lambda x: x[1], e['aspect_sentiment']))) > 1:
                    mixed.append(
                        {'sentence': e['sentence'], 'aspect': aspect, 'sentiment': sentiment})
                    mixed_counter[sentiment] += 1
                total_counter[sentiment] += 1
        print("total")
        print(total_counter)
        print("hard")
        print(mixed_counter)
        return unrolled, mixed

    @classmethod
    def splits_train_test(cls, text_field, as_field, sm_field, semeval_train, semeval_test, **kwargs):
        unrolled_train, mixed_train = SemEval.unroll(semeval_train)
        print("# Unrolled Train: {}    # Mixed Train: {}".format(len(unrolled_train), len(mixed_train)))

        unrolled_test, mixed_test = SemEval.unroll(semeval_test)
        print("# Unrolled Test: {}    # Mixed Test: {}".format(len(unrolled_test), len(mixed_test)))

        return (cls(text_field, as_field, sm_field, unrolled_train),
                cls(text_field, as_field, sm_field, unrolled_test),
                cls(text_field, as_field, sm_field, mixed_test))


class SemEval_TD(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, left_text_field, right_text_field, sm_field, input_data, **kwargs):

        text_field.preprocessing = data.Pipeline(clean_str)

        left_text_field.preprocessing = data.Pipeline(clean_str)
        left_text_field.init_token = '<beg>'

        right_text_field.preprocessing = data.Pipeline(clean_str)
        right_text_field.init_token = '<end>'

        fields = [('text', text_field), ('left_text', left_text_field), ('right_text', right_text_field),
                  ('sentiment', sm_field)]

        # unroll
        examples = []
        for e in input_data:
            examples.append(data.Example.fromlist([e['sentence'], e['left'], e['right'], e['sentiment']], fields))
        super(SemEval_TD, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def unroll(input_data):
        unrolled = []
        mixed = []
        for e in input_data:
            all_sentiments = set()
            for l, r, s in e['left_right']:
                unrolled.append({'sentence': e['sentence'], 'left': l, 'right': r, 'sentiment': s})
                all_sentiments.add(s)
            if len(all_sentiments) > 1:
                for l, r, s in e['left_right']:
                    mixed.append({'sentence': e['sentence'], 'left': l, 'right': r, 'sentiment': s})
        return unrolled, mixed

    @classmethod
    def splits(cls, text_field, left_text_field, right_text_field, sm_field, semeval_train, semeval_test, **kwargs):
        unrolled_train, mixed_train = SemEval_TD.unroll(semeval_train)
        print("# Unrolled Train: {}    # Mixed Train: {}".format(len(unrolled_train), len(mixed_train)))

        unrolled_test, mixed_test = SemEval_TD.unroll(semeval_test)
        print("# Unrolled Test: {}    # Mixed Test: {}".format(len(unrolled_test), len(mixed_test)))
        return (cls(text_field, left_text_field, right_text_field, sm_field, unrolled_train),
                cls(text_field, left_text_field, right_text_field, sm_field, unrolled_test),
                cls(text_field, left_text_field, right_text_field, sm_field, mixed_test))


class SemEval_RAN(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, offset_field, as_field, sm_field, input_data, **kwargs):
        """Create an SemEval dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            as_field: The field that will be used for aspect data.
            sm_field: The field that will be used for sentiment data.
            input_data: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('offset', offset_field), ('aspect', as_field), ('sentiment', sm_field)]

        # unroll the aspects of every sentence in the review.
        examples = []
        for e in input_data:
            examples.append(data.Example.fromlist([e['sentence'], e['of'], e['t'], e['sentiment']], fields))
        print("# Unrolled: {}".format(len(examples)))
        super(SemEval_RAN, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def unroll(input_data):
        unrolled = []
        mixed = []
        for e in input_data:
            all_sentiments = set()
            for of, t, s in e['offset']:
                unrolled.append({'sentence': e['sentence'], 'of': of, 't': t, 'sentiment': s})
                all_sentiments.add(s)
            if len(all_sentiments) > 1:
                for of, t, s in e['offset']:
                    mixed.append({'sentence': e['sentence'], 'of': of, 't': t, 'sentiment': s})
        return unrolled, mixed

    @classmethod
    def splits(cls, text_field, offset_field, as_field, sm_field, semeval_train, semeval_test, **kwargs):

        unrolled_train, mixed_train = SemEval_RAN.unroll(semeval_train)
        print("# Unrolled Train: {}    # Mixed Train: {}".format(len(unrolled_train), len(mixed_train)))

        unrolled_test, mixed_test = SemEval_RAN.unroll(semeval_test)
        print("# Unrolled Test: {}    # Mixed Test: {}".format(len(unrolled_test), len(mixed_test)))
        return (cls(text_field, offset_field, as_field, sm_field, unrolled_train),
                cls(text_field, offset_field, as_field, sm_field, unrolled_test),
                cls(text_field, offset_field, as_field, sm_field, mixed_test))
