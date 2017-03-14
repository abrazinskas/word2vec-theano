import itertools
import numpy as np
import glob
import nltk
import os
from utils import compute_distr, frequency_binning, allow_word
import src.settings as settings

# a general purpose vocabulary class
class Vocabulary:
    def __init__(self, train_data_path, vocab_file, maxsize=None, sep=' '):
        # creating vocabulary if it does not exist
        if not os.path.isfile(vocab_file):
            vocab = construct_vocabulary(train_data_path)
            write_vocabulary(vocab, vocab_file)
        dic = self.__read_vocabulary(vocab_file, maxsize, sep=sep)
        self.index_to_word = dic[0]
        self.word_to_index = dic[1]
        self.freq = dic[2]
        self.class_assignments = None
        self.class_boundaries = None
        self.class_to_words = None
        self.uni_distr = None
        self.ignored_token = settings.IGNORED_TOKEN

    # min_count: drop words from vocabulary that have count less than min_count
    # max_size: limit the vocabulary size (use only the top 'max_size' words in the vocabulary list)
    def __read_vocabulary(self, filename, maxsize=None, min_count=5, sep=' '):
        index_to_word = []
        freq = []
        with open(filename) as f:
            for word in itertools.islice(f, 0, maxsize):
                splt=word.split(sep)
                word = splt[0]
                count = int(splt[1])
                # dropping infrequent words
                if count >= min_count:
                    freq.append(count)
                    index_to_word.append(word)

        # appending special symbols
        index_to_word.append(settings.IGNORED_TOKEN)
        freq.append(0)
        index_to_word.append(settings.START_TOKEN)
        freq.append(0)
        index_to_word.append(settings.END_TOKEN)
        freq.append(0)

        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        return index_to_word, word_to_index, np.array(freq, dtype="float32")

    # words: a list of words
    def add_special_words(self, words):
        n = len(self.word_to_index)
        for i in range(len(words)):
            word = words[i]
            self.word_to_index[word] = n + i
            self.index_to_word.append(word)

    def assign_distr(self, pow=0.60):
        self.uni_distr = compute_distr(self.freq, pow)
        # dirty hack to avoid numpy's probabilies do not sum to 1
        s = sum(self.uni_distr)
        eps = 1 - s
        self.uni_distr[0]+=eps
        assert self.uni_distr is not None

    # performs frequency binning and assigns vocabulary items to bins
    def assign_classes(self, nr_classes):
        if self.uni_distr is None:
            self.assign_distr()
        self.class_assignments, self.class_boundaries, self.class_to_words = frequency_binning(self.uni_distr, nr_classes)
        assert self.class_assignments is not None

    def get_word(self, id):
        return self.index_to_word[id]

    def get_id(self, word):
        return self.word_to_index[word]

    def get_class(self, id):
        return self.class_assignments[id]

    def get_size(self):
        return len(self.word_to_index)
    def __len__(self):
        return len(self.index_to_word)



# works both for files and folders
def construct_vocabulary(loc):
    freqs = nltk.FreqDist()
    print("Creating vocabulary...")
    if os.path.isdir(loc):
        filenames = glob.glob(loc + "/*")
    else:
        filenames = [loc] # that means there is only one file
    for filename in filenames:
        with open(filename) as f:
            print("- Processing file " + filename + "...")
            words = nltk.word_tokenize(f.read().decode('utf-8').lower())
            words = [word for word in words if allow_word(word)]
            freqs.update(words)
    return freqs

def write_vocabulary(vocab, output_file, sep=' '):
    with open(output_file, 'w') as f:
        if isinstance(vocab, Vocabulary):
            f.write("\n".join([sep.join((str(word[0]), str(int(word[1])))) for word in zip(vocab.index_to_word, vocab.freq) if word[0] != settings.IGNORED_TOKEN]))
        else:
            f.write("\n".join([sep.join((word[0], str(word[1]))).encode('utf-8') for word in vocab.most_common()]))
    print("Vocabulary written to " + output_file)