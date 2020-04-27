# Introduction
This is an implementation of the paper [Solving Math Word Problems with Double-Decoder Transformer](https://arxiv.org/abs/1908.10924)

# How to use
## preparation
Clone this repository, and another repository for a [cusctomized transformer model in pytorch](https://github.com/text-machine-lab/transformerpy).
Use the **"stable" branch, not the "master" branch** to get results as in the paper.

You need the [Dolphin18k dataset](https://www.microsoft.com/en-us/research/project/sigmadolphin-2/), please contact me if you have difficulties in finding it.
We change the format of data a little bit. For example, one problem in the original dataset would be:
```
{
    "type": 0,
    "text": "one number is 4 less than another. the difference  of twice the smaller and 5 times the larger number is -11. What are the numbers?",
    "sources": [
      "https://answers.yahoo.com/question/index?qid=20080421132620AAXFXiG"
    ],
    "flag": 0,
    "original_text": "",
    "ans": "-3; 1",
    "equations": "unkn: x, y\r\nequ: x = y - 4\r\nequ: 2*x - 5*y = -11",
    "id": "yahoo.answers.20080421132620aaxfxig"
  }
```

In our form:
```
{
    "ans": "-3; 1",
    "text": "one number is 4 less than another. the difference  of twice the smaller and 5 times the larger number is -11. What are the numbers?",
    "id": "yahoo.answers.20080421132620aaxfxig",
    "unkn": "x,y",
    "equations": [
      "x=y-4",
      "2*x-5*y=-11"
    ]
  }
```
We parsed the equations into a list, and removed some unused information.

## config file
Go to your mathword folder, make changes in the config.py accordingly.
You do not need the NTM_PATH.

## Build data files for the model
Go to your mathword folder, run
```
python build_data.py -datafile 'your/data/file' -dest 'data/folder'
```
The data file should be in our format.

## Train model
To train a model with one decoder, run
```
python train.py -data data/folder/data.pt -save_model path/to/model -epoch 200 -save_mode interval
```
Check the source code to see more about the options.
To train a model with two decoders, run
```
python train.py -data data/folder/data.pt -save_model path/to/bi-model -epoch 200 -save_mode interval -bi
```
By default, the program will choose the first 80% as training data, and the rest as test data, 
in order to perform one experiment of the 5-fold cross validation.
Set the -offset option to 0.2 to use the 20-100% portion as training data; -offset 0.4 to use 40-100% plus 0-20%, and so on.

## Predict
To make predictions, run
```
python predict.py -model path/to/model -data data/folder/data.pt -output output.json -reset_num
```
if you used non-default -offset values for training, make sure you use the same here.
Othwerwise test data will overlap with training data.

## Compute score
```
python eval.py output.json
```
It will show the accuracy score.

## reinforcement training
```
python train_reinforce.py -model path/to/model -data data/folder/data.pt -batch_size 8 -save_model path/to/reinforcement/model -epochs 100
```
Here “path/to/model” is the one trained without reinforcement learning. It is your starting point.
Evaluation is the same as before.

