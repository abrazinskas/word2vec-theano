import theano
from theano import tensor as T
from src.word2vec import Word2Vec
from optimizations import SMOpt

class SkipGram(Word2Vec):
    def __init__(self, vocab, C=2, dim=100, lr_opt=None, sm_opt=SMOpt.none, nr_neg_samples=5):
        Word2Vec.__init__(self, vocab, C, dim, lr_opt, sm_opt, nr_neg_samples)
        self.__theano_build__()

    # x : context
    # y : center words
    def get_errors(self, x, y):
        H = self.V[y]
        if self.sm_opt == SMOpt.none:
            log_p_y_given_context = self.__sm_layer(H, x)
        elif self.sm_opt == SMOpt.negative_sampling:
            log_p_y_given_context = self.__ns_layer(H, x)
        return - log_p_y_given_context

    # for pure softmax output layer
    def __sm_layer(self, H, x):
        log_p_c = T.nnet.logsoftmax(T.dot(H, self.W))  # [b x v]
        log_p_c_given_w = log_p_c[T.arange(x.shape[0]).reshape((-1, 1)), x]
        return T.sum(log_p_c_given_w, axis=1)

    # negative sampling optimization
    def __ns_layer(self, H, x):
        b = x.shape[0]  # size of the batch
        neg_samples = self.sample(b)
        pos_repr = self.W[x.reshape((-1, ))]  # [b*2C x dim]
        H_1 = T.repeat(H, 2*self.C, axis=0)
        pos_scores = T.nnet.sigmoid(T.sum(pos_repr*H_1, axis=1))
        pos_scores = pos_scores.reshape((b, 2*self.C))

        neg_sample_matr = self.W[neg_samples]  # [k*b x dim ] (k - # of neg samples)
        H_2 = T.repeat(H, self.nr_neg_samples, axis=0)  # repeat each row k times [ b*k x n]
        neg_scores = T.nnet.sigmoid(T.sum(- H_2 * neg_sample_matr, axis=1))  # [b*k x 1]
        neg_scores = neg_scores.reshape((b, self.nr_neg_samples))  # [b x k]

        return T.sum(T.log(pos_scores), axis=1, dtype=theano.config.floatX) + T.sum(T.log(neg_scores), dtype=theano.config.floatX, axis=1)

