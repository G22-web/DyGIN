# Modality-Unified Dynamic Graph Inception Network for Context-Aware Emotion Recognition

<br>

**Apr. 17, 2025**
* First release of the project.

## Abstract
Graph representation learning has emerged as a widely researched topic, which derives effective node representations by recursively aggregating information from graph neighborhoods.
However, traditional methods for sequential data across various modalities, such as speech, facial expressions, and physiological signals, merely focus on a static graph constructed on an entire unimodal sequence, largely ignoring the dynamic evolution of the sequence itself. To overcome this limitation, we introduce a Modality-Unified Dynamic Graph Inception Network (DyGIN) that leverages dynamic evolutionary graphs across multiple modalities. First, these dynamic graphs are constructed successively on a series of subsequences segmented by a sliding window. Then, to enhance the temporal learning capabilities of the dynamic graphs, we employ a modified gated recurrent unit to update the graph weight at each segment. Additionally, we introduce a novel similarity matrix to update node representations instead of the traditional averaging method, where the matrix is calculated based on the degree of neighboring nodes. Finally, we combine graph structure loss, classification loss, and a learnable pooling function to jointly optimize the graph structure during training. To validate the effectiveness of the proposed model, we conducted experiments for emotion recognition across video, speech, and biosignal modalities using the RML, IEMOCAP, and DEAP datasets, respectively. Experimental results demonstrate that our model outperforms the latest (non)graph-based models. 

<img width="1055" alt="all" src="https://github.com/user-attachments/assets/b8edb777-0bd1-4c2a-8ac6-0748ba9b1189" />

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

