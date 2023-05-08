# CS-598 DL4H Reproducibility Project: Readmission Prediction via Deep Contextual Embedding of Clinical Concepts

This repository contains the code for this project that aimed to reproduce the result of the paper by Xiao et al. [Readmission Prediction via Deep Contextual Embedding of Clinical Concepts](https://doi.org/10.1371/journal.pone.0195024). 
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

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

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## How to train and test the model

1. Install all the dependencies
2. Save all the files from this repository under the same folder
![alt text](http://url/to/img.png)
3. Run Main.py
4. A prompt message will be shown on the terminal asking "Do you want to process the data? Enter Y or N: "
5. Enter "Y" if the data has never been processed before. The data needs to be processed before the first training
6. The data will be processed automatically. Then a prompt message will ask you "Please choose a model type:". If you want to train the CONTENT model, enter "CONTENT"; if you want to train the RNN model, enter "RNN"; if you want to train the CONTENT model with LSTM, enter "content_lstm"
7. The chosen model will be trained and tested. All the metrics will be printed to the terminal


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name				  | Top 1 Accuracy  | Top 5 Accuracy |
| --------------------------------------- |---------------- | -------------- |
| CONTENT (PyTorch, own implementation)   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
