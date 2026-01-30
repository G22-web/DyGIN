# Modality-Unified Dynamic Graph Inception Network for Context-Aware Emotion Recognition

<br>

## Abstract
Graph representation learning has attracted considerable attention for its ability to generate node representations by aggregating information from neighboring nodes. However, existing graph-based methods for sequential data merely focus on a static graph constructed on an entire unimodal sequence, largely ignoring their dynamic evolution. To overcome this limitation, we introduce a modality-unified **Dy**namic **G**raph **I**nception **N**etwork (**DyGIN**) that models dynamic evolutionary graphs for different modalities. DyGIN constructs dynamic graphs from temporal subsequences using a sliding window, and incorporates a Temporal Graph Evolution Gated Recurrent Unit (TGE-GRU) to update graph weights at each segment. Additionally, we propose a context-aware similarity matrix, which updates node representations based on the degree of neighboring nodes, replacing the traditional averaging method. Our model is optimized with a combination of graph structure loss, classification loss, and a learnable pooling function. We validate DyGIN on emotion recognition tasks across three public datasets—RML, IEMOCAP, and DEAP—where it outperforms state-of-the-art models by 1.6% on RML, 1.81% and 2.33% on IEMOCAP, and 2.78% and 6.81% on DEAP, based on accuracy and weighted F1 score. 

<img width="1107" alt="all" src="https://github.com/user-attachments/assets/b41d624c-b32e-4535-a1f0-a9458e9ce73b" />


<br>

## Preprocessing Data

The process for RML database is in preprocess directory. The process converts the database into one txt file including graph structure and node attributes.

Note: you can download the processed data from [here](https://drive.google.com/file/d/1KWygtpBUglY8gmzy0HuW6M8OL9u1V5sJ/view?usp=sharing) and put in this directory:

```
/dataset/
  Mine_Graph_RML/
    Mine_Graph_RML.txt
```


<br>

## Training

You can train a simple model with _main_DiceNew_ 


```
usage: main_DiceNew.py

optional arguments:
  -h, --help          Show this help message and exit
  -device             Which gpu to use if any
  -batch_size         Input batch size for training
  -iters_per_epoch    Number of iterations per each epoch
  -epochs             Number of epochs to train
  -lr                 Learning rate
  --num_layers        Number of inception layers
  --num_mlp_layers    Number of layers for MLP in each inception layer
  --hidden_dim        Number of hidden units for MLP
  --final_dropout     Dropout for classifier layer
  --Normalize         Normalizing data
  --patience          Patience for early stopping
```

<br>

