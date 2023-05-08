import pandas as pd
import csv
import pickle
import numpy as np
import math
import torch
from Hyperpara_config import Config as Config


def save_pkl(path, dump):
    with open(path, 'wb') as file:
        pickle.dump(dump, file)

# Embedding Defs
RARE_WORD = 100
STOP_WORD = 10000
UNKNOWN = 1

# Train/Validation Split Values
RAW_DATA_SIZE_SCALER = 1
RAW_DATA_SIZE = 3000
TRAIN_DATA_SIZE = math.floor(RAW_DATA_SIZE*RAW_DATA_SIZE_SCALER*2/3)
VALID_DATA_SIZE = math.floor(RAW_DATA_SIZE*RAW_DATA_SIZE_SCALER/5)
TEST_DATA_SIZE = int(RAW_DATA_SIZE*RAW_DATA_SIZE_SCALER)-TRAIN_DATA_SIZE-VALID_DATA_SIZE

'''
Sort Raw data by PID then Day_ID
Use this returned dataframe as the raw data
return:
  data     Dataframe       data from raw txt file
Data exmaple:
PID	    DAY_ID	DX_GROUP_DESCRIPTION	                        SERVICE_LOCATION	OP_DATE
1	        73546	LISINOPRIL	                                    Pharmacy_Claim	    74084
1	        73564	OTHER FORMS OF CHRONIC ISCHEMIC HEART DISEASE	DOCTORS OFFICE	    74084
1	        73564	OTHER POSTSURGICAL STATES	                    DOCTORS OFFICE	    74084
1	        73571	ANGINA PECTORIS	                                DOCTORS OFFICE	    74084
'''
def get_data(isSorted = False):
    # Read in the data from the input file
    if not isSorted:
        data = pd.read_csv(Config.RAW_DATA_PATH, sep="\t", header=0)
        data.to_csv(Config.RAW_DATA_SORTED_PATH, sep="\t", index=False)
        return data
    else:
        data = pd.read_csv(Config.RAW_DATA_PATH, sep="\t", header=0)
        # Sort the data by PID and DAY_ID
        sorted_data = data.sort_values(["PID", "DAY_ID"])
        data_size = int(RAW_DATA_SIZE*RAW_DATA_SIZE_SCALER)
        sorted_data = sorted_data[(sorted_data['PID'] >= 1) & (sorted_data['PID'] <= data_size)]
        # Write the sorted data to the output file
        sorted_data.to_csv(Config.RAW_DATA_SORTED_PATH, sep="\t", index=False)
        return sorted_data

'''
This function process the input data, then save it as CSV file
The description that appears more than RARE_WORD = 100 will be saved
It also sort the dataframe in ascending order, but not the stop word
It saves the stop word and vocab csv file.
Input:
  data,       Dataframe
  print_out,  boolean,    whether to print the details when running
Return:
  None,                   Save data to csv "./data/stop.txt", "./data/vocab.txt"
'''
def stop_vocab_generation(data):
    # Group Our Data By Description
    data_process = data.groupby('DX_GROUP_DESCRIPTION')
    # print("-----------<data_to_csv>----------")
    data_process = data_process.size() # add the frequency (size) of the groupby object
    data_process = data_process.to_frame('COUNT')
    data_process = data_process.reset_index()
    vocabDF = data_process[data_process['COUNT'] > RARE_WORD] #COUNT is the frequency of that description, it's the column name
    vocabDF = vocabDF.sort_values(by = 'COUNT').reset_index()['DX_GROUP_DESCRIPTION']
    vocabDF.index += 2 # it's essentially setting "unknown" to index 1.
    # example of rare before sorting
    #                                     DX_GROUP_DESCRIPTION  COUNT
    # 1                              ACQUIRED HYPOTHYROIDISM    27
    # 15                                     ANGINA PECTORIS    20
    # 18                                            ATENOLOL    13
    # 28                                CARDIAC DYSRHYTHMIAS    38
    # 33   CERTAIN ADVERSE EFFECTS, NOT ELSEWHERE CLASSIFIED    14
    # 34   CHRONIC AIRWAYS OBSTRUCTION, NOT ELSEWHERE CLA...    13
    # 47                               CLOPIDOGREL BISULFATE    16
    # 53                  DEFICIENCY OF B-COMPLEX COMPONENTS    12
    # 57                                   DIABETES MELLITUS   107
    # 59                             DILTIAZEM HYDROCHLORIDE    14
    # 68   DISORDERS OF FLUID, ELECTROLYTE, AND ACID-BASE...    13
    # 70                      DISORDERS OF LIPOID METABOLISM    55
    stop = data_process[data_process['COUNT'] > STOP_WORD]
    stop = stop.reset_index()['DX_GROUP_DESCRIPTION']
    
    vocabDF.to_csv(Config.VOCAB_PATH, sep = '\t', header = False, index = True)
    stop.to_csv(Config.STOP_PATH, sep = '\t', header = False, index = False)


'''
This function create a dictionary that stores the word-to-index mapping
It loads the vocab.txt file
Input:
  None
Return:
  word_to_index,    dict,      word-to-index mapping
sample output:
  {'apple': 2, 'banana': 3, 'orange': 4, 'pear': 5} start from 2 because .index+=2 above
'''
def load_vocab_index_dict():
    word_to_index = {}
    
    with open(Config.VOCAB_PATH, 'r') as vocab_file:
        read_in = csv.reader(vocab_file, delimiter='\t')
        # example of vocab file
        # 2	FRACTURE OF NECK OF FEMUR
        # 3	OTHER SYMPTOMS INVOLVING ABDOMEN AND PELVIS
        # 4	GENERAL MEDICAL EXAMINATION
        word_to_index = { entry[1]:int(entry[0]) for entry in read_in } 
        
    # Save Index to Word to Pickled File
    save_pkl(Config.VOCAB_PKL_PATH, {val:key for key, val in word_to_index.items()})
    
    return word_to_index

'''
This function extract all the 'INPATIENT HOSPITAL' event from the data file
grouby PID, DAY_ID, SERVICE_LOCATION, and add a count column
sort by PID then DAY_ID
return
  events, dataframe
'''
def extract_inpatient_events(print2checkBoolean = False):  
    # extract event "INPATIENT HOSPITAL"
    # target_event = 'INPATIENT HOSPITAL'

    df = pd.read_csv(Config.RAW_DATA_SORTED_PATH, sep='\t', header=0)
    events = df[df['SERVICE_LOCATION'] == 'INPATIENT HOSPITAL']
    events = events.groupby(['PID', 'DAY_ID', 'SERVICE_LOCATION']).size().to_frame('COUNT').reset_index()\
        .sort_values(by=['PID', 'DAY_ID'], ascending=True).set_index('PID')
    # events example
    # PID       DAY_ID  SERVICE_LOCATION        COUNT   
    # 1001      1       INPATIENT HOSPITAL      1
    # 1001      2       INPATIENT HOSPITAL      1
    # 1002      2       INPATIENT HOSPITAL      1
    # 1002      3       INPATIENT HOSPITAL      1

    if print2checkBoolean:
        # check event data
        events.to_csv('./data/event.txt',  sep = '\t', index = True)
        print("----Event extraction complete----")

    return events

'''
function:
  check whether if it has an hospital event in the next 30 days of the input PID
input:
  events = inpatient hospital event, dataframe
  pid = patient's PID, string/int
  day_id = string/int
return
  boolean, True=hospital event in the next 30 days
'''
def check_if_admitted_next30days(events, pid, day_id):
    try:                                # it uses try because some pid might be missing in events, those are out patient event
        patient = events.loc[int(pid)] #give a series/df of that patient.
        # if (int(pid) < 50):
        #     print(int(pid))
        #     print("<tag_logic> patient:")
        #     print((patient))
        # example of patient:
                # PID   DAY_ID    SERVICE_LOCATION  COUNT         
                # 1     73874  INPATIENT HOSPITAL      5
                # 1     73879  INPATIENT HOSPITAL      4
                # 1     73880  INPATIENT HOSPITAL      4
                # 1     74043  INPATIENT HOSPITAL      2

        # test whether have events within 30 days
        # check whether if it has an hospital event in the next 30 days, return TRUE it it does
        if isinstance(patient, pd.Series):
            if (int(day_id) <= patient.DAY_ID) & (patient.DAY_ID < int(day_id) + 30): # if <patient> is series not df (only 1 entry)
                return 1
            else:
                return 0
        # if dataframe, and no hospital in the next 30 days, it return -1 < 0: False
        if patient.loc[(int(day_id) <= patient.DAY_ID) & (patient.DAY_ID < int(day_id) + 30)].shape[0] > 0:
            return 1
        else:
            return 0
    except KeyError:
        # the label is not in the [index]
        # print("<tag_logic> keyError!")
        # print(f"in error!!!!!!!!!!!!!!!! Caused by patient {int(pid)}")
        return 0
    # print("======================you are in tag_logic")
    # patient = events.loc[int(pid)] # give a series of that patient
    # print("<tag_logic> patient:")
    # print(patient)


# 
# Input:
#   word_to_index,  dict
#   events,         dataframe
#   print_out,      boolean, 
# Return:
def seq_label_gen(word_to_index, events, print_out=False):
    # print("<convert_format_test> Running covert_format_test...")
    with open(Config.RAW_DATA_SORTED_PATH, mode='r') as f:
        # header
        header = f.readline().strip().split('\t')
        # print(f"<convert_format_test> header: {header}")
        if print_out:
            print(header)
        pos = {}
        for key, value in enumerate(header):    # the first line of the txt file is header
            pos[value] = key                    # {'PID': 0, 'DAY_ID': 1, 'DX_GROUP_DESCRIPTION': 2, 'SERVICE_LOCATION': 3, 'OP_DATE': 4}
        if print_out:
            print(pos)
        # print(f"<convert_format_test> pos: {pos}")

        # when each patient is complete, label is appended to labels.
        # So labels and docs have the size of num of patients.
        # label = size(num_patient, num different day_id]
        # docs = size(num_patient, num different day_id, num of events]

        docs = []# it appends doc every time patient id changes
        doc = [] # it appends sent every time date id changes
        sent = [] # sent is the DX_Group_description mapped integer, every line
        labels = [] # appends every time patient id changes
        label = [] # appends every time check_if_admitted_next30days is ran (patient id changed, day id changed), each patient has a label<list>, size of number of different dayID of that patient
        # label saves whether the dayID has an admission in the next 30 days.

        # init
        line = f.readline() # header is already read, now it's reading the content
        tokens = line.strip().split('\t') # this gives a list of strings ["1", "73888", "ANGINA PECTORIS", "DOCTORS OFFICE", "74084"]
        pid = tokens[pos['PID']]    # 1
        day_id = tokens[pos['DAY_ID']] # 73888
        label.append(check_if_admitted_next30days(events, pid, day_id)) #event:dataframe, pid:string, day_id:string
        # print("<convert_format_test> reading the first line:")
        # print(f"<convert_format_test> line: {line}")
        # print(f"<convert_format_test> tokens: {tokens}")
        # print(f"<convert_format_test> pid: {pid}")
        # print(f"<convert_format_test> day_id: {day_id}")
        # print(f"<convert_format_test> label: {label}")
        # print(f"<convert_format_test> -----------------last print before while loop------------------")
        while line != '':
            tokens = line.strip().split('\t')
            c_pid = tokens[pos['PID']]
            c_day_id = tokens[pos['DAY_ID']]

            # when c_pid or c_dayid changed, it runs check_if_admitted_next30days()
            # closure
            if c_pid != pid:        # pid is the last pid that ran check_if_admitted_next30days(), c_pid is current pid
                # print(f"<convert_format_test> --------------------in c_pid != pid-------------")
                # print(f"<convert_format_test> pid: {pid}")
                # print(f"<convert_format_test> day_id: {day_id}")
                # print(f"<convert_format_test> new cpid: {c_pid}")
                # print(f"<convert_format_test> new c_day_id: {c_day_id}")
                # print(f"<convert_format_test> line: {line}")
                # print(f"<convert_format_test> tokens: {tokens}")
                # print(f"<convert_format_test> label: {label}")
                # print(f"<convert_format_test> doc: {doc}")
                # print(f"<convert_format_test> docs: {docs}")
                # print(f"<convert_format_test> sent: {sent}")
                # print(f"<convert_format_test> labels: {labels}")
                # print(f"<convert_format_test> label: {label}")
                # print(f"--------------------during computing for the current line-------------")
                doc.append(sent)
                docs.append(doc)
                # print(f"<convert_format_test> doc (updated): {doc}")
                # print(f"<convert_format_test> docs (updated): {docs}")
                sent = []
                doc = []
                pid = c_pid
                day_id = c_day_id
                labels.append(label)
                label = [check_if_admitted_next30days(events, pid, day_id)]
                # print(f"--------------------after computing for the current line-------------")
                # print(f"<convert_format_test> in c_pid != pid")
                # print(f"<convert_format_test> pid (updated): {pid}")
                # print(f"<convert_format_test> day_id (updated): {day_id}")
                # print(f"<convert_format_test> cpid: {c_pid}")
                # print(f"<convert_format_test> c_day_id: {c_day_id}")
                # print(f"<convert_format_test> line: {line}")
                # print(f"<convert_format_test> tokens: {tokens}")
                # print(f"<convert_format_test> doc (updated): {doc}")
                # print(f"<convert_format_test> docs (updated): {docs}")
                # print(f"<convert_format_test> sent (updated): {sent}")
                # print(f"<convert_format_test> labels (updated): {labels}")
                # print(f"<convert_format_test> label (updated): {label}")
                # print(f"--------------------------------------------------------------------")
            else:
                if c_day_id != day_id:
                    # print(f"<convert_format_test> --------------------c_day_id != day_id-------------")
                    doc.append(sent)
                    sent = []
                    day_id = c_day_id
                    label.append(check_if_admitted_next30days(events, pid, day_id))
            #         print(f"<convert_format_test> pid: {pid}")
            #         print(f"<convert_format_test> cpid: {c_pid}")
            #         print(f"<convert_format_test> c_day_id (updated): {c_day_id}")
            #         print(f"<convert_format_test> line: {line}")
            #         print(f"<convert_format_test> tokens: {tokens}")
            #         print(f"<convert_format_test> doc (updated): {doc}")
            #         print(f"<convert_format_test> docs: {docs}")
            #         print(f"<convert_format_test> sent (updated): {sent}")
            #         print(f"<convert_format_test> labels: {labels}")
            #         print(f"<convert_format_test> label (updated): {label}")
            #         print(f"--------------------------------------------------------------------")
            # print(f"--------------------------Outside of pid dayid comparison------------------------------------------")
            word = tokens[pos['DX_GROUP_DESCRIPTION']] # get the description
            try:
                sent.append(word_to_index[word]) # word_to_index only has the words in vocab.txt, the popular words, this line maps the index and save it in {sent}
                # print(f"<convert_format_test> appending to sent: {word_to_index[word]}")
            except KeyError:
                sent.append(UNKNOWN) # all the not popular words are "unknown", it's mapped as index 1
                # print(f"<convert_format_test> appending to sent: {UNKNOWN}")
            # print(f"<convert_format_test> sent (updated): {sent}")
            line = f.readline()
            # print(f"<convert_format_test> line (updated): {line}")
            # print(f"--------------------------------------------------------------------")

        # closure
        doc.append(sent)
        docs.append(doc)
        labels.append(label)
        # print("<convert_format_test> label at the end:")
        # print(label)
        # print("<convert_format_test> label length:")
        # print(len(label))

    return docs, labels

def splits(X, labels):
    save_pkl(Config.X_TRAIN_PATH, X[:TRAIN_DATA_SIZE])
    save_pkl(Config.X_VALID_PATH, X[TRAIN_DATA_SIZE:(TRAIN_DATA_SIZE + VALID_DATA_SIZE)])
    save_pkl(Config.X_TEST_PATH,  X[TRAIN_DATA_SIZE + VALID_DATA_SIZE:])
    save_pkl(Config.Y_TRAIN_PATH, labels[:TRAIN_DATA_SIZE])
    save_pkl(Config.Y_VALID_PATH, labels[TRAIN_DATA_SIZE:(TRAIN_DATA_SIZE + VALID_DATA_SIZE)])
    save_pkl(Config.Y_TEST_PATH,  labels[TRAIN_DATA_SIZE + VALID_DATA_SIZE:])


# This prepare_data function is downloaded from https://github.com/danicaxiao/CONTENT/blob/master/CONTENT.py
def prepare_data(seqs, labels, vocabsize, maxlen=None):
    """Create the matrices from the datasets.
    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.
    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    eventSeq = []

    for seq in seqs:
        t = []
        for visit in seq:
            t.extend(visit)
        eventSeq.append(t)
    eventLengths = [len(s) for s in eventSeq]

    if maxlen is not None:
        new_seqs = []
        new_lengths = []
        new_labels = []
        for l, s, la in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_lengths.append(l)
                new_labels.append(la)
            else:
                new_seqs.append(s[:maxlen])
                new_lengths.append(maxlen)
                new_labels.append(la[:maxlen])
        lengths = new_lengths
        seqs = new_seqs
        labels = new_labels

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = torch.zeros((n_samples, maxlen, vocabsize))
    x_mask = torch.zeros((n_samples, maxlen))
    y = torch.ones((n_samples, maxlen))
    for idx, s in enumerate(seqs):
        x_mask[idx, :lengths[idx]] = 1
        for j, sj in enumerate(s):
            for tsj in sj:
                x[idx, j, tsj-1] = 1
    for idx, t in enumerate(labels):
        y[idx,:lengths[idx]] = torch.tensor(t)

    return x, x_mask, y, lengths, eventLengths