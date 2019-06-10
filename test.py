import argparse
import gensim.models.fasttext as fasttext
import torch
import torch.nn as nn

from utils import load_model, Testset

if __name__ != '__main__':
  raise Exception('This file cannot be imported.')

# create parser
parser = argparse.ArgumentParser(description='KynG (testing script)')

# device option
parser.add_argument('-g', '--gpu', action='store_true', help='train on gpu if available')

# loading option
parser.add_argument('-em', '--embedding', metavar='<file>', type=str, default='fastText.bin', help='fastText embedding trained by official fastText implementation(default=fastText.bin)')
parser.add_argument('--eos', metavar='<str>', type=str, default='<eos>', help='symbol for end of sentence(default=<eos>)')
parser.add_argument('-m', '--model', metavar='<path>', type=str, default='KynG.pt', help='path to rained model(default=KynG.pt)')

# generation option
parser.add_argument('-d', '--data', metavar='<file>', type=str, default='testing.txt', help='file that contains keyword(default=testing.txt)')
parser.add_argument('-l', '--len', metavar='<int>', type=int, default=20, help='length of text generated(default=20)')
parser.add_argument('-n', '--num', metavar='<int>', type=int, default=5, help='number of text to create for each keyword(default=5)')
parser.add_argument('-r', '--result', metavar='<path>', type=str, default='KynG.txt', help='file to save generation results(default=KynG.txt)')

# parse arguments
args = parser.parse_args()

# device
on_gpu = args.gpu and torch.cuda.is_available()
device = torch.device('cuda' if on_gpu else 'cpu')

# model
print(f'Loading model from {args.model}')
netG = load_model(args.model, device)

# hyper-parameters
emb_dim = netG.emb_dim
n_vocabs = netG.n_vocabs

# index to word
print(f'Loading embedding from {args.embedding}')
gensim_emb = fasttext.load_facebook_vectors(args.embedding)
ind2word = gensim_emb.index2entity

# guard for same embedding while training
assert(emb_dim == gensim_emb.vector_size)
assert(n_vocabs == len(gensim_emb.vocab))

# pytorch embedding
torch_emb = nn.Embedding(n_vocabs, emb_dim)
torch_emb.weight.data.copy_(torch.tensor(gensim_emb.vectors))
torch_emb = torch_emb.to(device)

# testing data
print(f'Loading data from {args.data}')
test_set = Testset(args.data, gensim_emb)

# create text
out_stream = open(args.result, 'w')

# creating text
total = args.num * len(test_set)
num_generated = 0

with torch.no_grad():
  for k, e in test_set:
    out_stream.write(f'keyword: {k}\n')

    # batch size of 1
    e = e.unsqueeze(0).to(device)

    for i in range(args.num):
      generated = netG(e, args.len, torch_emb)
      word_idx = generated.squeeze(0).cpu().max(1)[1]
      words = []
      for idx in word_idx:
        word = ind2word[idx]
        if word == args.eos:
          break
        words.append(word)
      text = ' '.join(words)

      out_stream.write(f'{i}: {text}\n')
      num_generated += 1
      print(f'\rText {num_generated}/{total}', end = '', flush = True)

out_stream.close()
print(f'\nGenerated text saved at {args.result}')
