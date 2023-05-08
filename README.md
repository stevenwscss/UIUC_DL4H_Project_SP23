# CS-598 DL4H Reproducibility Project: Readmission Prediction via Deep Contextual Embedding of Clinical Concepts

This repository contains the code for this project that aimed to reproduce the result of the paper by Xiao et al. [Readmission Prediction via Deep Contextual Embedding of Clinical Concepts](https://doi.org/10.1371/journal.pone.0195024). 

## Reference Materials
This reproducibility project used the source code from Xiao et al. The source code of the paper can be found on the GitHub repository, [https://github.com/danicaxiao/CONTENT](https://github.com/danicaxiao/CONTENT). The data used in this project is also from the paper, which can be downloaded from [https://ndownloader.figstatic.com/files/11029691](https://ndownloader.figstatic.com/files/11029691).

## Data Download Instruction
The data used in this project is also from the paper, which can be downloaded from [https://ndownloader.figstatic.com/files/11029691](https://ndownloader.figstatic.com/files/11029691). Unzip the downloaded "txtData.zip" file. The raw data is named as "S1_File.txt". This is the file that we used to train and test the model.

## Requirements

We used the following dependencies:
1. os
2. itertools
3. numpy
4. math
5. pickle
6. csv
7. PyTorch
8. scikit-learn

To install the dependencies, use the follow command:

```setup
pip install -r requirements.txt
```

## How to train and test the model

1. Install all the dependencies
2. Save all the files from this repository under the same folder
![alt text](http://url/to/img.png)
3. Run `Main.py`
4. A prompt message will be shown on the terminal asking `Do you want to process the data? Enter Y or N:`
5. Enter "Y" if the data has never been processed before. The data needs to be processed before the first training.
6. The data will be processed automatically. Then a prompt message will ask you `Please choose a model type: `. If you want to train the CONTENT model, enter "CONTENT'"; if you want to train the RNN model, enter "RNN"; if you want to train the CONTENT model with LSTM, enter "content_lstm".
7. The chosen model will be trained and tested. All the metrics will be printed to the terminal.

## Results

Our model achieves the following performance on:

| Model name				  | ROC-AUC	    | PR-AUC	     | ACC	     |
| --------------------------------------- |---------------- | -------------- | --------------|
| CONTENT (PyTorch, own implementation)   |0.7998±0.0014    | 0.6501±0.0018  | 0.8352±0.0001 |
| CONTENT (reported in paper)  		  |0.6886±0.0074    | 0.6011±0.0191  | 0.7170±0.0069 |
| GRU (reported in paper)   		  |0.6881±0.0048    | 0.5929±0.0100  | 0.7141±0.0040 |
| GRU (own implementation)  		  |0.7937±0.0003    | 0.6445±0.0012  | 0.8318±0.0017 |
| CONTENT w/ LSTM (own implementation)    |0.7937±0.0024    | 0.6440±0.0003  | 0.8320±0.0010 |



## References

Cao Xiao, Tengfei Ma, Adji B. Dieng, David M. Blei, and Fei Wang. 2018. [Readmission prediction via deep contextual embedding of clinical concepts](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0195024). PLOS ONE, 13:e0195024.

Cao Xiao, Tengfei Ma, Adji B. Dieng, David M. Blei, and Fei Wang. (2018). CONTENT (Version 1.0.0) [Computer software].https://doi.org/10.1371/journal.pone.0195024
