# importing modules

import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torch.utils.data import DataLoader
from torch import nn
import os
import io
import time
import csv
import torch
import torch.onnx as onnx
import torchvision.models as models

import pickle
import joblib


#URL dict containing train and test dataset name
train_rows=0
test_rows=0
with open("train_italian.csv") as f:
	train_rows=sum(1 for line in f)

with open("test_italian.csv") as f:
	test_rows=sum(1 for line in f)



URL = {
    'train':"train_italian.csv" ,
    'test': "test_italian.csv",
}

#specify number of lines
NUM_LINES = {
    'train': train_rows,
    'test': test_rows,
}

#This is overrided method it takes parameter as train or test convert into another
#and return the object
@_wrap_split_argument(('train', 'test'))
def AG_NEWS(root, split):
    def _create_data_from_csv(data_path):
        with io.open(path, encoding="utf8", errors='ignore') as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])

    path = URL[split]
    return _RawTextIterableDataset("AG_NEWS", NUM_LINES[split],
                                   _create_data_from_csv(path))

#It is very basic components of the torchtext library,
# including vocab, word vectors, tokenizer.
# Those are the basic data processing building blocks for raw text string.


tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))

vocab = Vocab(counter, min_freq=1)

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: int(x) - 1


# Before sending to the model, collate_fn function works on
# a batch of samples generated from DataLoader.
# The input to collate_fn is a batch of data with the batch size in DataLoader,
# and collate_fn processes them according to the data processing pipelines
# declared previouly. Pay attention here and make sure that collate_fn is declared
# as a top level def. This ensures that the function is available in each worker.


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

#Loading training data

train_iter = AG_NEWS(split='train')
#Calling DataLoader built in class giving batch size, training data and cleaned data
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)


#Defining the model
# The model is composed of the nn.EmbeddingBag layer plus a linear layer
# for the classification purpose. nn.EmbeddingBag with the default mode of “mean”
# computes the mean value of a “bag” of embeddings.
# Although the text entries here have different lengths, nn.EmbeddingBag module
# requires no padding here since the text lengths are saved in offsets.

# Additionally, since nn.EmbeddingBag accumulates
# the average across the embeddings on the fly, nn.EmbeddingBag can
# enhance the performance and memory efficiency to process a sequence of tensors


class TextClassificationModel(nn.Module):

    #takesm vocab size,dimensiona and number of class (4)
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    #definig weights for the system
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    #taking each line of dataset
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)




# Split the dataset and run the model

# Since the original AG_NEWS has no valid dataset,
# we split the training dataset into train/valid sets with a split ratio of 0.95
# (train) and 0.05 (valid).
# Here we use torch.utils.data.dataset.random_split function in PyTorch core library
#
# CrossEntropyLoss criterion combines nn.LogSoftmax() and nn.NLLLoss() in a single class. It is useful when training a classification problem with C classes. SGD implements stochastic gradient descent method as the optimizer. The initial learning rate is set to 5.0.
# StepLR is used here to adjust the learning rate through epochs.




train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class)

#train the model,calculating total time to train,specifying accuracy
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
#evaulating the model ,checking accuracy measure
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

from torch.utils.data import Dataset, DataLoader, random_split
# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

#specify batch_size,epochs, apply optimzer to reduce loss,spliting training and testing data
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter = AG_NEWS(split='train')
test_iter=AG_NEWS(split='test')
train_dataset = list(train_iter)
test_dataset = list(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ =random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

#Looping according to given epochs,calculate time of each epoch ,and measure the accuracy
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test*100))







#saving and downloading the model
torch.save(model, "model.pt")

# files.download("model.pth")


