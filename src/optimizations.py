from enum import Enum

# softmax optimizations
class SMOpt(Enum):
    none = 1
    negative_sampling = 2
    hierarchical_softmax = 3 # NOT a huffman's tree

# learning rate optimizations
class LROpt:
    def __init__(self, learning_rate):
        self.alpha = learning_rate

class Adam(LROpt):
    def __init__(self,learning_rate, beta1, beta2):
        LROpt.__init__(self, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2

class SGD(LROpt):
    def __init__(self, learning_rate):
        LROpt.__init__(self, learning_rate)

class AdaGrad(LROpt):
    def __init__(self):
        raise NotImplementedError