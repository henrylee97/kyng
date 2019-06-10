import argparse
import gensim.models.fasttext as fasttext
import os
import numpy as np
import pickle
import torch
import torch.nn
import torch.nn.functional as f

from datetime import datetime
from model import *
from torch.utils.data import DataLoader
from utils import Trainset

if __name__ != '__main__':
  raise Exception('This file cannot be imported.')

# create parser
parser = argparse.ArgumentParser(description='KynG (training script)')

# device option
parser.add_argument('-g', '--gpu', action='store_true', help='train on gpu if available')

# preloading option
parser.add_argument('-em', '--embedding', metavar='<file>', type=str, default='fastText.bin', help='fastText embedding trained by official fastText implementation(default=fastText.bin)')
parser.add_argument('-d', '--data', metavar='<csv>', type=str, default='training.csv', help='csv that contains training data(default=training.csv)')
parser.add_argument('--num-workers', metavar='<int>', type=int, default=1, help='number of workers for preparing data(default=1)')

# generator option
parser.add_argument('-rs', '--rand-size', metavar='<int>', type=int, default=30, help='size of random in generator(default=30)')
parser.add_argument('-hg', '--hidden-g', metavar='<int>', type=int, default=500, help='size of hidden layer in lstm in generator(default=500)')
parser.add_argument('-lg', '--layer-g', metavar='<int>', type=int, default=2, help='number of layers in lstm in generator(default=2)')
parser.add_argument('-dg', '--dropout-g', metavar='<float>', type=float, default=0.5, help='dropout rate of generator(default=0.5)')

# rnn discriminator option
parser.add_argument('-hd', '--hidden-d', metavar='<int>', type=int, default=500, help='size of hidden layer in lstm in discriminator(default=500)')
parser.add_argument('-ld', '--layer-d', metavar='<int>', type=int, default=2, help='number of layers in lstm in discriminator(default=2)')
parser.add_argument('-dd', '--dropout-d', metavar='<float>', type=float, default=0.5, help='dropout rate of discriminator(default=0.5)')

# cnn discriminator option
parser.add_argument('--cnn', action='store_true', help='use cnn based discriminator(default=rnn based discriminator)')
parser.add_argument('--filter', metavar='<int>', type=int, default=100, help='size of filter in convolution layer(default=100)')
parser.add_argument('--windows', metavar='<int>', type=int, default=[3,4,5], nargs='+', help='size of window of convolution layer(default=3 4 5)')

# training option
parser.add_argument('-bs', '--batch-size', metavar='<int>', type=int, default=100, help='size of batch(default=100)')
parser.add_argument('-e', '--epoch', metavar='<int>', type=int, default=200, help='training epochs(default=200)')
parser.add_argument('-lrg', '--learning-rate-g', metavar='<float>', type=float, default=0.0000001, help='learning rate of generator(default=0.0000001)')
parser.add_argument('-lrd', '--learning-rate-d', metavar='<float>', type=float, default=0.0000001, help='learning rate of discriminator(default=0.0000001)')
parser.add_argument('-tf', '--teach-force', metavar='<float>', type=float, default=0.5, help='teach force ratio(default=0.5)')

# save model option
parser.add_argument('-m', '--model', metavar='<path>', type=str, default='KynG', help='path to save trained model; .pt will be added(default=KynG)')
parser.add_argument('--save-epoch', metavar='<path>', type=str, default=None, help='save every epoch in folder(default=disabled)')
parser.add_argument('--save-discriminator', action='store_true', help='save discriminator')
parser.add_argument('--save-history', action='store_true', help='save history')

# parse arguments
args = parser.parse_args()

if args.save_epoch is not None:
  os.makedirs(args.save_epoch, exist_ok = True)

# device setting
on_gpu = args.gpu and torch.cuda.is_available()
device = torch.device('cuda' if on_gpu else 'cpu')

# load embedding
# loading gensim embedding
print(f'Loading embedding from {args.embedding}')
gensim_emb = fasttext.load_facebook_vectors(args.embedding)
emb_dim = gensim_emb.vector_size
n_vocabs = len(gensim_emb.vocab)

# make torch embedding for generator
torch_emb = nn.Embedding(n_vocabs, emb_dim)
torch_emb.weight.data.copy_(torch.tensor(gensim_emb.vectors))
torch_emb.require_grad = False  # disable update
torch_emb = torch_emb.to(device)

# make torch linear embedding for discriminator
linear_emb = nn.Linear(n_vocabs, emb_dim, bias = False)
linear_emb.weight.data.copy_(torch.tensor(gensim_emb.vectors).t())
# linear_emb.require_grad = False # disable update
linear_emb = linear_emb.to(device)

# load data
print(f'Loading data from {args.data}')
dataset = Trainset(args.data, gensim_emb)
train_set = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, drop_last = True)

# create models
print('Create models')

netG = Generator(emb_dim, n_vocabs, args.rand_size, args.hidden_g, args.layer_g, args.dropout_g, device).to(device)

if args.cnn:
  netD = CNNDiscriminator(emb_dim, args.filter, args.windows, args.dropout_d).to(device)
else:
  netD = RNNDiscriminator(emb_dim, args.hidden_d, args.layer_d, args.dropout_d, device).to(device)

optD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate_d)
optG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate_g)

# training
total_batch = len(train_set)
history = {'args': args, 'D': [], 'G': []}

y_real, y_fake = torch.ones(args.batch_size).to(device), torch.zeros(args.batch_size).to(device)
criterionD = nn.BCELoss().to(device)
criterionK = nn.MSELoss().to(device)

start_time = datetime.now()
print(f'Training started at {start_time}')

netD.train()
netG.train()

# training epoch
for e in range(args.epoch):
  D_losses = []
  G_losses = []

  # training total training set
  for i, batch in enumerate(train_set):

    keyword = batch['keyword'].to(device)
    text = batch['text'].to(device)
    length = batch['length'].to(device)

    # sort batch in descending order
    length, sorted_idx = length.sort(0, descending = True)
    keyword = keyword[sorted_idx]
    text = text[sorted_idx]

    # train D

    optD.zero_grad()
    linear_emb.zero_grad()

    # real -> real
    D_real, K_real = netD(text, length)
    D_loss_real = criterionD(D_real, y_real) + criterionK(K_real, keyword)
    D_loss_real.backward()

    # fake -> fake
    generated = netG(keyword, length.max(), torch_emb, text, args.teach_force)

    gen_text = F.softmax(generated, dim = 1)
    gen_text = linear_emb(gen_text)

    D_fake, K_fake = netD(gen_text.detach(), length)
    D_loss_fake = criterionD(D_fake, y_fake)
    D_loss_fake.backward()

    D_loss = D_loss_real + D_loss_fake

    optD.step()

    # update G

    optG.zero_grad()
    
    # fake -> real
    D_gen, K_gen = netD(gen_text, length)
    G_loss = criterionD(D_gen, y_real) + criterionK(K_gen, keyword)
    G_loss.backward()

    optG.step()

    D_loss = np.mean(D_loss.item())
    G_loss = np.mean(G_loss.item())
    D_losses.append(D_loss)
    G_losses.append(G_loss)

    print(f'\rEpoch {e + 1}/{args.epoch}, Step {i + 1}/{total_batch}, D loss: {D_loss:.6f}, G loss: {G_loss:.6f}', end = '', flush = True)

  history['D'].append(D_losses)
  history['G'].append(G_losses)

  print(f'\rEpoch {e + 1}/{args.epoch}, Step {total_batch}/{total_batch}, Average D loss: {np.mean(D_losses):.6f}, Average G loss: {np.mean(G_losses):.6f}')

  if args.save_epoch is not None:
    netG.save(os.path.join(args.save_epoch, f'e{e + 1}_G.pt'))
    if args.save_discriminator:
      netD.save(os.path.join(args.save_epoch, f'e{e + 1}_D.pt'))

end_time = datetime.now()
print(f'Training ended at {end_time}')
print(f'Total training time: {end_time - start_time}')
print(f'D loss: {np.mean(D_losses)}, G loss: {np.mean(G_losses)}')

netG.save(f'{args.model}.pt')
if args.save_discriminator:
  netD.save(f'{args.model}_D.pt')

if args.save_history:
  with open(f'{args.model}_history.pkl', 'wb') as f:
    pickle.dump(history, f)
