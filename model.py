import torch
import torch.nn as nn
import torch.nn.functional as F

# one-to-many rnn
# generate text from a keyword
class Generator(nn.Module):
  def __init__(self, emb_dim, n_vocabs, rand_size = 300, hidden_size = 500, num_layers = 2, dropout = 0.5, device = torch.device('cpu')):
    super(Generator, self).__init__()

    # initial parameters
    self.emb_dim = emb_dim
    self.n_vocabs = n_vocabs
    self.rand_size = rand_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout_rate = dropout
    self.device = device

    # layers
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(emb_dim + rand_size, hidden_size, num_layers, batch_first = True, bidirectional = False, dropout = dropout)
    self.fc = nn.Linear(hidden_size, n_vocabs)

  # save
  def save(self, save_dir):
    torch.save({
      'class': 'RNN-G',
      'emb_dim': self.emb_dim,
      'n_vocabs': self.n_vocabs,
      'rand_size': self.rand_size,
      'hidden_size': self.hidden_size,
      'num_layers': self.num_layers,
      'dropout': self.dropout_rate,
      'state_dict': self.state_dict()
    }, save_dir)

  # initial hidden memory and cell
  def _init_hidden(self, batch_size = 1):
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    return h0, c0

  # forwarding
  def forward(self, keyword, maxlen, embedding, target = None, teach_force = 0.5):
    # keyword     : initial keyword
    # maxlen      : maximun length for generation
    # embedding   : pretrained word embedding that contains self.n_vocabs words
    # target      : target text
    # teach_force : teach force ratio while training

    batch_size = keyword.size(0)

    # if target is None, we cannot use force teaching
    if target is None:
      teach_force = 0.0
    
    # output container: [B, L, V]
    output = torch.zeros(batch_size, maxlen, self.n_vocabs).to(self.device)

    # initial hidden state
    hidden = self._init_hidden(batch_size)

    # forwarding words
    # initial random input
    z = torch.cat((keyword, torch.randn(batch_size, self.rand_size).to(self.device)), -1)

    for i in range(maxlen):

      # generate output
      z = z.unsqueeze(1)
      z = self.dropout(z)
      out, hidden = self.lstm(z, hidden)
      out = out.squeeze(1)
      out = self.fc(out)
      output[:, i] = out

      # create next word embedding
      out = embedding(out.max(1)[1]) if torch.rand(1) > teach_force else target[:, i]
      z = torch.cat((out, torch.randn(batch_size, self.rand_size).to(self.device)), -1)

    return output

# many-to-one rnn
# discriminate the text is real or fake
# our own structure
#   real or fake + keyword
class RNNDiscriminator(nn.Module):
  def __init__(self, emb_dim, hidden_size = 500, num_layers = 2, dropout = 0.5, device = torch.device('cpu')):
    super(RNNDiscriminator, self).__init__()

    # initial parameters
    self.emb_dim = emb_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout_rate = dropout
    self.device = device

    # layers
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, bidirectional = True, batch_first = True, dropout = dropout)
    self.fc = nn.Linear(hidden_size * 2, 1 + emb_dim)

  # save
  def save(self, save_dir):
    torch.save({
      'class': 'RNN-D',
      'emb_dim': self.emb_dim,
      'hidden_size': self.hidden_size,
      'num_layers': self.num_layers,
      'dropout': self.dropout_rate,
      'state_dict':self.state_dict()
    }, save_dir)

  # inital memory and cell
  def _init_hidden(self, batch_size = 1):
    h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
    return h0, c0

  # forwarding
  def forward(self, text, length):
    # text   : batch of texts that sorted in descending order of their length
    # length : lengths of texts in batch

    batch_size = text.size(0)

    # dropout
    text = self.dropout(text)

    # pack text with length
    text = nn.utils.rnn.pack_padded_sequence(text, length, batch_first = True)

    # forward text
    _, (hidden, _) = self.lstm(text, self._init_hidden(batch_size))
    out = torch.cat([h for h in hidden[-2:]], 1)
    out = self.fc(out)

    # return real/fake, extracted keyword from text
    return torch.sigmoid(out[:, 0]), out[:, 1:]

# cnn
# discriminate the text is real or fake
# our own structure
#   real or fake + keyword
class CNNDiscriminator(nn.Module):
  def __init__(self, emb_dim, n_filter = 100, window_sizes = [3, 4, 5], dropout = 0.5):
    super(CNNDiscriminator, self).__init__()

    # initial parameters
    self.emb_dim = emb_dim
    self.n_filter = n_filter
    self.window_sizes = window_sizes
    self.dropout_rate = dropout

    # layers
    self.convs = nn.ModuleList([nn.Conv1d(in_channels = emb_dim, out_channels = n_filter, kernel_size = ws) for ws in window_sizes])
    self.fc = nn.Linear(len(window_sizes) * n_filter, 1 + emb_dim)
    self.dropout = nn.Dropout(dropout)
  
  # save
  def save(self, save_dir):
    torch.save({
      'class': 'CNN-D',
      'emb_dim': self.emb_dim,
      'n_filter': self.n_filter,
      'window_sizes': self.window_sizes,
      'dropout': self.dropout_rate,
      'state_dict': self.state_dict()
    }, save_dir)

  # forwarding
  def forward(self, text, *args, **kwargs):
    # text         : batch of texts that sorted in descending order of their length
    # args, kwargs : due to compatibility with RNNDiscriminator

    # text = [B, T, E] -> [B, E, T]
    text = text.permute(0, 2, 1)
    
    # network
    conved = [F.relu(conv(x)) for conv in self.convs]
    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
    cat = self.dropout(torch.cat(pooled, dim = 1))
    out = self.fc(cat)

    # return real/fake, extracted keyword from text
    return torch.sigmoid(out[:, 0]), out[:, 1:]
