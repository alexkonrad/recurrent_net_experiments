import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import BinaryAddition, CharacterDataset, EchoKthNumber
from torch.utils.data import DataLoader
import gru


class BaselineModel(pl.LightningModule):
  def __init__(self, input_dim, state_dim, output_dim, many_to_one=True):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.gru = nn.GRU(self.input_dim, state_dim, batch_first=True)
    self.output_layer = nn.Linear(state_dim, self.output_dim)
    self.loss_fn = nn.CrossEntropyLoss()
    self.training_losses = []
    self.many_to_one = many_to_one

  def forward(self, x):
    out, h = self.gru(x)
    if self.many_to_one:
      out = out[:1, -1, :]
    output_layer_activations = self.output_layer(out)
    return output_layer_activations

  def training_step(self, batch, batch_idx):
    x, y = batch
    pred = self(x)
    pred = pred.reshape(-1, self.output_dim)
    y = y.flatten()
    loss = self.loss_fn(pred, y)
    self.training_losses.append(loss.item())
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    pred = self.forward(x)
    loss = self.loss_fn(pred, y)
    return loss

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=0.01)

  def on_epoch_end(self):
    train_loss_mean = np.mean(self.training_losses)
    self.logger.experiment.add_scalar('training_loss', train_loss_mean, global_step=self.current_epoch)
    self.training_losses = []
    print("[epoch %d/%d] Avg. Loss for last 500 samples = %lf" % (self.current_epoch + 1, 20, train_loss_mean))


# Training parameters
epochs = 20
iterations = 500

# Echo Kth Number dataset
# input_dim = 10
# state_size = 20
# output_dim = 10
# kth = 2
# min_seq_len = 3
# max_seq_len = 8
# train_dataset = EchoKthNumber(iterations, kth, min_seq_len, max_seq_len)
# train_loader = DataLoader(train_dataset)
# model = BaselineModel(input_dim, state_size, output_dim, many_to_one=True)

# Code for training Character-RNN
# epochs = 5
# input_dim = 64
# state_size = 20
# output_dim = 64
# train_dataset = CharacterDataset()
# train_loader = DataLoader(train_dataset, batch_size=200, collate_fn=CharacterDataset.collate_fn)
# model = BaselineModel(input_dim, state_size, output_dim, many_to_one=False)


# Code for binary addition RNN
epochs = 5
input_dim = 2
q_hidden_size = 10
p_hidden_size = 12
output_dim = 1
train_dataset = BinaryAddition(2000)
train_loader = DataLoader(train_dataset, batch_size=2)
model = BaselineModel(input_dim, state_size, output_dim, many_to_one=False)

trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=iterations)

trainer.fit(model, train_loader)

print("Finished training")
