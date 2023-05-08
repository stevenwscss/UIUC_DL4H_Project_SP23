# CS-598 DL4H Reproducibility Project: Readmission Prediction via Deep Contextual Embedding of Clinical Concepts

This repository contains the code for this project that aimed to reproduce the result of the paper by Xiao et al.[Readmission Prediction via Deep Contextual Embedding of Clinical Concepts](https://doi.org/10.1371/journal.pone.0195024). 
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Reference Materials
This reproducibility project used the source code from Xiao et al. The source code of the paper can be found in [this](https://github.com/danicaxiao/CONTENT) GitHub repository. The data used in this project is also from the paper, which can be downloaded [here](https://ndownloader.figstatic.com/files/11029691).

## Data Download Instruction
The data used in this project is also from the paper, which can be downloaded [here](https://ndownloader.figstatic.com/files/11029691). Unzip the downloaded "txtData.zip" file. The raw data is named as "S1_File.txt". This is the file that we used to train and test the model.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
