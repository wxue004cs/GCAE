import gensim
import numpy as np
import linecache


def load_w2v_embedding(word_list, uniform_scale, dimension_size):
    embed_file = '../../../code/embedding/GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=True)
    word_vectors = []
    for word in word_list:
        if word in model:
            word_vectors.append(model[word])
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors


def load_glove_embedding(word_list, uniform_scale, dimension_size):
    glove_words = []
    glove_file = '../../Glove Vectors/glove.840B.300d.txt'
    with open(glove_file, 'r') as fopen:
        for line in fopen:
            line_split = line.split()
            word = line_split[:-300]
            word = ' '.join(word)
            glove_words.append(word)
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(glove_file, word2offset[word]+1)
            line_split = line.split()
            line_word = line_split[:-300]
            line_word = ' '.join(line_word)
            assert(word == line_word)
            word_vectors.append(np.array(line_split[-300:], dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors


def load_yelp_embedding(word_list, uniform_scale, dimension_size):
    word2embed = {}
    with open('../../../code/embedding/yelp_embedding.txt', 'r') as fopen:
        for line in fopen:
            w = line.split(sep=' ')
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
    word_vectors = []
    c = 0
    for word in word_list:
        if word in word2embed:
            c += 1
            s = np.array(word2embed[word], dtype=np.float32)
            word_vectors.append(s)
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    print('Yelp initializes {}'.format(c))
    return word_vectors


def save_embedding(word_list, word_embedding,
                   word_list_file='embedding/yelp_words.txt',
                   word_embedding_file='embedding/yelp_embedding.txt'):
    with open(word_list_file, 'w') as fopen:
        for w in word_list:
            fopen.write(w + '\n')

    with open(word_embedding_file, 'w') as fopen:
        for i in range(len(word_list)):
            w = word_list[i]
            fopen.write(w)
            for n in word_embedding[i]:
                fopen.write(' {:.5f}'.format(n))
            fopen.write('\n')


def load_aspect_embedding_from_w2v(aspect_list, word_stoi, w2v):
    aspect_vectors = []
    for w in aspect_list:
        aspect_vectors.append(w2v[word_stoi[w.split()[0]]])
    return aspect_vectors


def load_aspect_embedding_from_file(aspect_list, file_path):
    aspect_vectors = {}
    d = 0
    with open(file_path, 'r') as fopen:
        for line in fopen:
            w, v = line.split(':')
            v = np.fromstring(v, sep=' ')
            d = len(v)
            aspect_vectors[w] = v
    vecs = []
    for a in aspect_list:
        if a not in aspect_vectors:
            vecs.append(np.random.uniform(-0.25, 0.25, d))
        else:
            vecs.append(aspect_vectors[a.lower()])
    return vecs, d

if __name__ == '__main__':
    pass