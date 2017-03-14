import os
import glob
import numpy as np
import re
import errno
from evaluation.GloVe.evaluate import glove_evaluate
from evaluation.word_sim.all_wordsim import all_word_sim
from evaluation.word_sim.wordsim import word_sim
import src.settings as settings
SUBSAMPLING_THRESHOLD = 10e-5



# Sub-sampling of frequent words: can improve both accuracy and speed for large data sets
# Source: "Distributed Representations of Words and Phrases and their Compositionality"
def allow_with_prob(word_count, total_words_count):
    freq = float(word_count) / float(total_words_count)
    removal_prob = 1.0 - np.sqrt(SUBSAMPLING_THRESHOLD / freq)
    return np.random.random_sample() > removal_prob


# if word_count and total_count are provided, will perform sub-sampling
def allow_word(word, vocab=None, subsample=False):
    if subsample and vocab is not None:
        return allow_with_prob(vocab.get_count(word), vocab.total_count)
    return True


# Tokenizes the files in the given folder
# - Converts to lower case, removes punctuation
# - Yields sentences split up in words, represented by vocabulary index
# Files in data folder should be tokenized by sentence (one sentence per newline),
# Like in the 1B-words benchmark
def tokenize_files(vocab_dict, data_path, subsample_frequent=False):
    # total_dict_words = sum([value for key, value in vocab_dict.iteritems()])
    # detect if the path with data is actually a file
    if os.path.isdir(data_path):
       filenames = glob.glob(data_path + "/*")
    else:
        filenames = [data_path] # that means there is only one file
    for filename in filenames:
        with open(filename) as f:
            for sentence in f:
                # Use nltk tokenizer to split the sentence into words or a standard way (faster)
                words = re.sub(ur"\p{P}+", "", sentence.decode('utf-8').lower()).split()
                #words =  word_tokenize(sentence.decode('utf-8').lower())
                # Filter (remove punctuation)
                tokens = [(vocab_dict[word] if word in vocab_dict else vocab_dict[settings.IGNORED_TOKEN]) for word in words if allow_word(word)]
                # Replace words that are not in vocabulary with special token
                # words = [word if word in vocab_dict else IGNORED_TOKEN for word in words]
                # Yield the sentence as indices
                if words:
                    yield tokens

def create_context_windows(sentence, window_size):
    n = len(sentence)
    for idx in range(window_size, n - window_size):
        context = sentence[idx - window_size:idx] + sentence[idx + 1:idx + window_size + 1]
        yield (sentence[idx], context)

def files_len(folder):
    filenames = glob.glob(folder + "/*")
    for fname in filenames:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
    return i + 1


# evaluates vectors on Glove's benchmark and offline wordvectors.org benchmark
# vectors_path : filename or a folder with vectors
def evaluate(vectors_path, vocab_file):
    if os.path.isdir(vectors_path):
        filenames = glob.glob(vectors_path + "/*")
    else:
        filenames = [vectors_path] # that means there is only one file

    for filename in filenames:
        # https://github.com/mfaruqui/eval-word-vectors
        all_sim_file = "evaluation/word_sim/data/combined-word-sim/TEST.txt"
        word_sim(filename, all_sim_file)
        sim_folder = "evaluation/word_sim/data/word-sim"
        all_word_sim(filename, sim_folder)
        # https://github.com/stanfordnlp/GloVe
        glove_evaluate(vocab_file, filename)


# computes and returns a unigram distributions over frequency counts
def compute_distr(freq, pow=0.75):
    p = freq**pow
    return p/np.sum(p,dtype="float32")


# distr: a valid distribution (e.g. over words)
# we are making an assumption that the ids of the words correspond to the order of provided in the distr
def frequency_binning(distr, nr_bins):
    assert round(sum(distr),3) == 1
    prob_per_bin = 1.0/nr_bins
    class_boundaries = [0]
    class_assignments = []
    class_to_words = [[]]
    bin = 0
    cur_bin_prob = 0
    for i in range(len(distr)):
        pr = distr[i]
        # check if it's time to change the bin as it's full and if the item can't fit into the bin
        # we make an exception for the last bin, because some bins can be non-full
        if (bin != nr_bins-1) and (cur_bin_prob >= prob_per_bin or pr > prob_per_bin - cur_bin_prob):
            bin += 1
            cur_bin_prob = 0
            class_boundaries.append(i)
            class_to_words.append([])
        # assign the item to the bin
        class_assignments.append(bin)
        cur_bin_prob += pr
        class_to_words[bin].append(i)
    class_boundaries.append(len(distr))
    return class_assignments, class_boundaries, class_to_words


# loads from data_folder chunks of data of specified size
# it is useful for mini-batch gradient methods
# vocab : vocabulary object
# restrict : if set to True will not include instances that have smaller than 2*window elements,
#            the implication is that we will not have corner words
# preload : if True it will preload the data into the main memory
def load_data_chunks(data_folder, vocab, chunk_size, window_size=2, restrict=True, preload=False):
    if preload:
        print 'pre-loading sentences into the main memory'
        sentences = []
        temp_sentences = tokenize_files(vocab.word_to_index, data_folder)
        for sentence in temp_sentences:
            sentences.append(sentence)
        print 'done'
    else:
        sentences = tokenize_files(vocab.word_to_index, data_folder)

    # create data placeholders
    X = []
    y = []
    i = 0 # counter
    for sent in sentences:
        sent = prepend_start_end(sent, vocab, window_size)  # to avoid chopping off corner words
        cws = create_context_windows(sent, window_size)
        for cw in cws:
            if len(cw[1]) < 1 or (restrict and len(cw[1]) < 2*window_size):
                continue
            X.append(cw[1])
            # here we assign to y_ either just an id of the word or the (id, class)
            y_ = cw[0]
            y.append(y_)
            i += 1
            # return the chunk when the container gets full
            if i >= chunk_size:
                yield X, y
                # reset
                i = 0
                X = []
                y = []
    if len(X) > 0:
        yield X, y


# prepend start and end symbols to the sentence
def prepend_start_end(sentence, vocab, window_size):
    return [vocab.get_id(settings.START_TOKEN)]*window_size + sentence + [vocab.get_id(settings.END_TOKEN)]*window_size


# for float comparison
def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def create_folders_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise