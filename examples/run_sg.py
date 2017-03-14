from tools.support import train_model, print_params
from tools.vocabulary import Vocabulary, write_vocabulary
from collections import OrderedDict
from src.skipgram import SkipGram, SMOpt
from tools.utils import evaluate
from src.optimizations import Adam, SGD

train_data_path = '../data/training/10M'
loss_data_path = '../data/training/1M'
vocab_file = '../data/vocabulary/10M.txt'
vectors_folder = 'output/vectors/10M/'
temp_vocab_file = "output/vocabulary/vocab.txt"

###### PARAMETERS ######
C = 5  # window size (one sided)
n = 100  # the number of components in the hidden layer
alpha = 0.0015  # learning rate
sm_opt = SMOpt.negative_sampling
epochs = 1
neg_samples = 16
max_vocab_size = None
batch_size = 500
save_loss = True
preload = False  # preload data into the main memory

vocab = Vocabulary(train_data_path=train_data_path, vocab_file=vocab_file, maxsize=max_vocab_size)
if sm_opt == SMOpt.negative_sampling:
    vocab.assign_distr(pow=0.75)
lr_opt = Adam(alpha, beta1=0.9, beta2=0.999)
print_params(OrderedDict((('model', 'SG'), ('context_window_size', C), ('hidden_layer', n),
                         ('learning_rate', alpha), ('epochs', epochs), ('batch_size', batch_size),
                         ('train_data_path', train_data_path), ('vocabulary_size', len(vocab)))))
model = SkipGram(vocab=vocab, C=C, dim=n, sm_opt=sm_opt, lr_opt=lr_opt, nr_neg_samples=neg_samples)
train_model(model, train_data_path=train_data_path, loss_data_path=loss_data_path, vocab=vocab, window=C,
                     chunk_size=batch_size, epochs=epochs, save_loss=save_loss, preload=preload)

model.save_word_vectors(vectors_folder)
write_vocabulary(vocab, temp_vocab_file)  # it's necessary to save vocab for accuracy estimation
evaluate(vectors_folder, temp_vocab_file)
