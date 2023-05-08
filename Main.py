import os
import time
import DataProcessing as dp
from Hyperpara_config import Config
from PatientDataLoader import Data_Loader
import PyTorch_CONTENT


start_time = time.time()
script_dir = os.path.dirname(os.path.abspath(__file__)) # get script directory
os.chdir(script_dir) # change the directory

def DataProcessing(isSorted = False):
    print("------------Loading and processing data------------")
    start_time = time.time()
    print("Loading and sorting raw data......")
    data = dp.get_data(isSorted) # load the raw data
    print("Creating Stop and Vocab files......")
    dp.stop_vocab_generation(data) # Writes vocab.txt and stop.txt, using data. vocab.txt contains all the description that has appearance more than x
    print("Mapping vocab to index......")
    word_index_dict = dp.load_vocab_index_dict() # Save a vocab.pkl file, it's a dict
    print("Extracting in patient events......")
    eventsDF = dp.extract_inpatient_events() # Extracts all the inpatient events, it's a dataframe
    # print(events)
    print("Processing data to sequence and labels......")
    seq, labels = dp.seq_label_gen(word_index_dict, eventsDF) # create the sequence and labels for training
    print("Splitting data into training, testing and validation sets......")
    dp.splits(seq, labels) # splits the training, validation and testing sets
    print("Total data processing time: {:.3f}s".format(time.time() - start_time))
    print("------------Complete loading and processing data------------")

# Ask user whether to process data
answer = input("Do you want to process the data? Enter Y or N: ")
# Check user's input
if answer.lower() == 'y':
    # Process the data
    DataProcessing()
else:
    print("Data processing skipped.")

FLAGS = Config()
data_set = Data_Loader(FLAGS)
iterator = data_set.iterator()

model_type = None
while model_type not in ['rnn', 'content', 'content_lstm']:
    model_type = input("Please choose a model type:\n 'rnn' for RNN (GRU)\n 'content' for CONTENT\n 'content_lstm' for CONTENT (LSTM)\n").lower()
    if model_type == 'rnn':
        isRNN = True
        isCONTENT = False
        isLSTM = False
    elif model_type == 'content':
        isRNN = False
        isCONTENT = True
        isLSTM = False
    elif model_type == 'content_lstm':
        isRNN = False
        isCONTENT = False
        isLSTM = True
    else:
        print("Invalid input. Please choose again.")
PyTorch_CONTENT.run(data_set, FLAGS, isCONTENT, isRNN, isLSTM)