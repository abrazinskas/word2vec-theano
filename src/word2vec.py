import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from optimizations import Adam, SGD, AdaGrad
from tools.utils import create_folders_if_not_exist
import lasagne
import settings
from optimizations import SMOpt


theano.config.floatX = 'float32'
theano.optimizer_including='cudnn'
np.random.seed(42)


# base class
class Word2Vec():
    # vocab : a vocabulary object
    # C : context window (one sided)
    # dim : the number of components in the hidden layer
    # lr_opt : learning rate optimisation object
    # neg_samples : the number of negative samples
    def __init__(self, vocab, C=2, dim=100, lr_opt=None, sm_opt=SMOpt.none, nr_neg_samples=5):
        self.v = len(vocab.index_to_word)
        self.vocab = vocab
        self.lr_opt = lr_opt
        self.dim = dim
        self.C = C  # size of one side of the window
        self.sm_opt = sm_opt # softmax optimization type

        # weights initialization
        self.V = theano.shared(value=np.float32(np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(self.v, dim)))
                       ,name='V')  # input representation matrix
        self.W = theano.shared(value=np.float32(np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(self.dim, self.v)))
                               , name='W') # output representation matrix
        if sm_opt==SMOpt.negative_sampling:
            self.p = theano.shared(vocab.uni_distr)
            self.W = theano.shared(value=np.zeros(shape=(self.v, self.dim), dtype=theano.config.floatX), name='W') # output representation matrix
            if self.p is None:
                raise RuntimeError('Please provide a probability distribution over words')
            self.nr_neg_samples = nr_neg_samples
        self.params = [self.V, self.W]

    def __theano_build__(self):
        x = T.imatrix(name='x')  # context words
        y = T.ivector(name='y')  # center words
        errors = self.get_errors(x, y)
        mean_error = T.mean(errors, dtype=theano.config.floatX)
        unnorm_error = T.sum(errors)
        updates = self.get_updates(mean_error)
        self.train = theano.function(inputs=[x, y], outputs=mean_error, updates=updates)

        # some extra support functions
        self.mean_error = theano.function(inputs=[x, y], outputs=mean_error)
        self.unnorm_error = theano.function(inputs=[x, y], outputs=unnorm_error)
        word_idx = T.iscalar('word_idx')
        out_rep = self.W[word_idx, :] if self.sm_opt == SMOpt.negative_sampling else self.W[:, word_idx]
        self.get_word_input_rep = theano.function(inputs=[word_idx], outputs=self.V[word_idx, :])
        self.get_word_output_rep = theano.function(inputs=[word_idx], outputs=out_rep)
        self.repr_types= {"input":self.get_word_input_rep, "output":self.get_word_output_rep}

    # returns an update dictionary based on the learning rate optimization method
    def get_updates(self, cost):
        if isinstance(self.lr_opt, Adam):
            updates = lasagne.updates.adam(cost, self.params, learning_rate=self.lr_opt.alpha, beta1=self.lr_opt.beta1,
                                           beta2=self.lr_opt.beta2)
        elif isinstance(self.lr_opt, AdaGrad):
            raise NotImplementedError
        else:  # SGD
            updates = lasagne.updates.sgd(cost, self.params, learning_rate=self.lr_opt.alpha)
        return updates

    # x: is a matrix of context words
    # y: is a vector of target words
    def calculate_total_loss(self, x, y):
        return self.unnorm_error(x, y)

    def calculate_total_loss_mean(self, x, y):
        return self.mean_error(x, y)

    # used by negative sampling
    def sample(self, b):
        r = RandomStreams(seed=1)
        return r.choice(size=(self.nr_neg_samples*b,), replace=True, a=self.v, p=self.p, dtype='int32')

    def save_word_vectors(self, vectors_folder):
        for name, func in self.repr_types.items():
            self.___save_vectors(self.vocab.index_to_word, vectors_folder+name+".txt", func)

    # saves wordvectors into a file
    def ___save_vectors(self, index_to_word, file, embeddings_function):
        create_folders_if_not_exist(file)
        print 'writing word vectors to %s' % file
        with open(file, 'w') as output_file:
            for idx, word in enumerate(index_to_word):
                if word == settings.IGNORED_TOKEN:
                    continue
                word_vec = embeddings_function(idx)
                output_file.write(word + " " + " ".join(str(f) for f in word_vec)+"\n")