'''
Some hyperparameters are not defined by the paper;
So we define them with our best judgement based on the experiment results.
'''
class Config:
    dataset = ""
    data_path = "./%s" % dataset
    n_topics = 50
    learning_rate = 0.001
    vocab_size = 619
    n_stops = 22 
    lda_vocab_size = vocab_size - n_stops
    n_hidden = 200
    embed_dim = 100
    total_epoch = 5
    batch_size = 1
    threshold = 0.5
    # File Paths
    VOCAB_PATH = "./vocab.txt"
    STOP_PATH = "./stop.txt"
    VOCAB_PKL_PATH = "./vocab.pkl"
    RAW_DATA_PATH = "./S1_File.txt"
    RAW_DATA_SORTED_PATH = "./raw_data_sorted.txt"
    X_TRAIN_PATH = "./X_train.pkl"
    X_VALID_PATH = "./X_valid.pkl"
    X_TEST_PATH = "./X_test.pkl"
    Y_TRAIN_PATH = "./Y_train.pkl"
    Y_VALID_PATH = "./Y_valid.pkl"
    Y_TEST_PATH = "./Y_test.pkl"