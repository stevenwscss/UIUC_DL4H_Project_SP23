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
    data = dp.get_data(isSorted) # Retrieves Input Data If it doesn't already exist, return pd.dataframe
    print("Creating Stop and Vocab files......")
    dp.stop_vocab_generation(data) # Writes vocab.txt and stop.txt, using data. vocab.txt contains all the description that has appearance more than x
    print("Mapping vocab to index......")
    word_index_dict = dp.load_vocab_index_dict() # Loads vocab from csv files into word2ind vector, also save a vocab.pkl file, it's a dict
    print("Extracting in patient events......")
    eventsDF = dp.extract_inpatient_events() # Extracts Events From Input File, a dataframe
    # print(events)
    print("Processing data to sequence and labels......")
    seq, labels = dp.seq_label_gen(word_index_dict, eventsDF) # Converts Data into More Useful Format
    print("Splitting data into training, testing and validation sets......")
    dp.splits(seq, labels) # Creates Training, Validation, and Testing Splits and Writes them to Pickled Files
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




# notes for myself to understand the code label and docs, please ignore when grading
# Each patient has a labels<list>
# tag is ran when pid!=c_pid
# label is saved to labels, when pid changes

# Each group description is converted into integer by mapping, and store in sent<list>
# Then sent will be appended to doc, when pid or dayID change


# Each same patient has many doc<list>:
# [[31, 1, 46], [1, 52, 51, 45], [1], [1, 1, 1], [51, 36, 45, 1], [12], [26], [1]]
# Each list inside is from the same day_id, and from sent<list> to doc<list> when the day changes
# At each patient change, doc is appended to docs and reset, like labels.

# label size: num of day_id change (aka num of differen day_id in a patient xx)
# each patient has one label<list>
# label of patient 1:
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# when each patient is complete, label is appended to labels.
# So labels and docs have the size of num of patients.
# label = size(num_patient, num different day_id), whether admission happen in the next 30 days, will be used as y?
# docs = size(num_patient, num different day_id, num of events] what are the events for each patient, will be used as x? doesn't seem to be sequential
# docs is a 3d list