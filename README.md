# KynG: Keyword &amp; Generator
Text generator for Korean

## Clone
via ssh
```
$ git clone git@github.com:HenryLee97/KynG.git
```
via https
```
$ git clone https://github.com/HenryLee97/KynG.git
```

## Requirement
Python 3.6 or above  
Install [PyTorch](https://pytorch.org)
```
$ pip install gensim
```

## Training

### Need
* pretrained fastText embedding, trained by [fastText](https://github.com/facebookresearch/fastText)
* csv file, formated as follows:
```
keyword, text
```

### Usage
Use ```train.py``` to train model. For example,
```
$ python train.py -g -em fastText.bin -d training.csv -m KynG -e 10
```
Detailed options can be checked as follows:
```
$ python train.py -h
```

## Testing

### Need
* pretrained fastText embedding, which used for training
* text file, that contains one keyword in each row.

### Usage
Use ```test.py``` to test trained model. For example,
```
$ python test.py -g -em fastText.bin -d testing.txt -m KynG.pt -r KynG.txt 
```
Detailed options can be checked as follows:
```
$ python test.py -h
```
