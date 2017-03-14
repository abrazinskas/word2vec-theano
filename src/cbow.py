import theano
from theano import tensor as T
from src.word2vec import Word2Vec
from optimizations import SMOpt

# Continues bag of words class
class CBOW(Word2Vec):
    def __init__(self, vocab, C=2, dim=100, lr_opt=None, sm_opt=SMOpt.none, nr_neg_samples=5):
        Word2Vec.__init__(self, vocab, C, dim, lr_opt, sm_opt, nr_neg_samples)
        self.__theano_build__()

    # x : context words
    # y: center words
    def get_errors(self, x, y):
        H = T.mean(self.V[x], axis=1, dtype=theano.config.floatX)
        if self.sm_opt == SMOpt.none:
            log_p_y_given_context = self.__sm_layer(H, y)
        elif self.sm_opt == SMOpt.negative_sampling:
            log_p_y_given_context = self.__ns_layer(H, y)  # this is not really a probability over words
        return - log_p_y_given_context

    # for pure softmax output layer
    def __sm_layer(self, H, y):
        log_p_y = T.nnet.logsoftmax(T.dot(H, self.W))  # [b x v]
        return log_p_y[T.arange(y.shape[0]), y]  # extract only the relevant words for the loglikelihood

    # negative sampling
    def __ns_layer(self, H, y):
        b = y.shape[0]  # size of the batch
        neg_samples = self.sample(b)
        pos_scores = T.nnet.sigmoid(T.sum(self.W[y]*H, axis=1))  # [b x 1]
        H1 = T.repeat(H, self.nr_neg_samples, axis=0)
        neg_scores = T.nnet.sigmoid(T.sum(- H1 * self.W[neg_samples], axis=1))  # [b*k x 1]
        neg_scores = neg_scores.reshape((b, self.nr_neg_samples))  # [b x k]
        return T.log(pos_scores) + T.sum(T.log(neg_scores), dtype=theano.config.floatX, axis=1)