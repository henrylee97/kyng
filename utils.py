import csv
import torch

from model import *
from torch.utils.data import Dataset

# training dataset for own text
class Trainset(Dataset):
  def __init__(self, filename, embedding):
    # filename  : csv file that contains data
    # embedding : pretrained gensim fastText embedding

    # initial parameters
    self.filename = filename
    self.embedding = embedding

    # load all data
    self.data = []
    with open(filename) as f:
      cr = csv.reader(f)
      for line in cr:
        self.data.append((line[0], line[1].split()))

    # maximum length of text
    maxlen = 0
    for _, t in self.data:
      maxlen = max(maxlen, len(t))

    # metavar
    self.len = len(self.data)
    self.emb_dim = embedding.vector_size
    self.maxlen = maxlen

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    keyword = self.data[idx][0]
    text = self.data[idx][0]

    # keyword embedding
    keyword = torch.tensor(self.embedding[keyword])

    # text embedding
    text = torch.stack([torch.tensor(self.embedding[text[i]]) if i < len(text) else torch.zeros(self.emb_dim) for i in range(self.maxlen)])

    return {'keyword': keyword, 'text': text, 'length': torch.tensor(len(text))}

# testing dataset for own data
class Testset:
  def __init__(self, filename, embedding):
    # filename  : text file that contains keyword in each row
    # embedding : pretrained gensim fastText embedding

    # initial parameters
    self.filename = filename
    self.embedding = embedding

    # load all data
    self.data = []
    with open(filename) as f:
      for line in f:
        self.data.append(line.strip())
    
    # metavar
    self.len = len(self.data)
  
  def __len__(self):
    return self.len

  def __iter__(self):
    self.current = 0
    return self
  
  def __next__(self):
    self.current += 1
    if self.current > self.len:
      raise StopIteration
    else:
      k = self.data[self.current - 1]
      return k, torch.tensor(self.embedding[k])

# model loading method
def load_model(load_dir, device = torch.device('cpu')):
  d = torch.load(load_dir)
  
  # check model class
  # if Generator
  if d['class'] == 'RNN-G':
    model = Generator(d['emb_dim'], d['n_vocabs'], d['rand_size'], d['hidden_size'], d['num_layers'], d['dropout'], device)
    model.load_state_dict(d['state_dict'])
    return model.to(device)
  
  # if RNN based Discriminator 
  elif d['class'] == 'RNN-D':
    model = RNNDiscriminator(d['emb_dim'], d['hidden_size'], d['num_layers'], d['dropout'], device)
    model.load_state_dict(d['state_dict'])
    return model.to(device)

  # if CNN based Discriminator
  elif d['class'] == 'CNN-D':
    model = CNNDiscriminator(d['emb_dim'], d['n_filter'], d['window_sizes'], d['dropout'], device)
    model.load_state_dict(d['state_dict'])
    return model.to(device)

  # unknown class
  else:
    raise Exception('Class ({}) is unknown.'.format(d['class']))
