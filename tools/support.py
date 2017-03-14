# import matplotlib.pyplot as plt
import time
from utils import load_data_chunks
from src.cbow import SMOpt

def train_model(model, train_data_path, loss_data_path, vocab, epochs=1, save_loss=False, window=5, chunk_size=500, preload=False):
    losses = []
    if save_loss:
        l = compute_loss(model, vocab, loss_data_path, window=window, chunk_size=chunk_size)
        losses.append(l)
        print("LOSS OF %s is %f " % (loss_data_path, l))
    start = time.time()
    # TODO: make preload only once
    for epoch in range(epochs):
        counter = 0
        print("--- EPOCH : %d ---" % (epoch+1))
        chunks = load_data_chunks(train_data_path, vocab, chunk_size=chunk_size, window_size=window, classes=(model.sm_opt==SMOpt.hierarchical_softmax), preload=preload)
        for chunk in chunks:
            counter += 1
            temp_loss = model.train(chunk[0], chunk[1])
            if counter % 1000 == 0 : print "chunk's # %d loss: %f" % (counter, temp_loss)
        if save_loss:
            l = compute_loss(model, vocab, loss_data_path, window=window, chunk_size=chunk_size)
            losses.append(l)
            print("LOSS OF %s is %f " % (loss_data_path, l))
    end = time.time()
    print "training took %f minutes" % ((end - start)/60.0)
    return losses


# computes the average cross entropy loss of the model
# data has to be in the format X - matrix of context windows, y - vector with target words
def compute_loss(model, word_to_index, data_path, window=5, chunk_size=500):
    loss = 0.0
    word_num = 0.0
    counter = 0
    chunks = load_data_chunks(data_path, word_to_index, chunk_size=chunk_size, window_size=window)
    for chunk in chunks:
        counter += 1
        # print("processing chunk %d" % counter)
        word_num += len(chunk[1])
        loss += model.calculate_total_loss(chunk[0], chunk[1])
    return loss/word_num


def print_params(params):
    print '----------------------------'
    print '---- EXPERIMENT\'S SETUP ----'
    for param_name, param_value in params.iteritems():
        print param_name + ": " + str(param_value)
    print '--------------------------'


