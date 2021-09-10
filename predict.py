# importing modules

import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torch
import torch.onnx as onnx
import torchvision.models as models
from torchtext.utils import download_from_url, unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
from torch.utils.data import DataLoader
import os
import io
from torch import nn


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






#It is very basic components of the torchtext library,
# including vocab, word vectors, tokenizer.
# Those are the basic data processing building blocks for raw text string.

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

#make tokens of training dataset
tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))

vocab = Vocab(counter, min_freq=1)
#The text and label pipelines will be
# used to process the raw data strings from the dataset iterators.

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]



#making new Class TextClassificationModel
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




#reading labels from file


r=open("labels.txt",encoding="utf-8")
ag_news_label ={}
for i in r.readlines():
    data=i.strip().split(",")
    ag_news_label[int(data[0])]=data[1]


#defining predict function take text as return the predicted label having highest
#accuracy
def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


#loading the model
model = torch.load("model.pt")

model.eval()
inp=input("Enter the string ")

predict_label=predict(inp, text_pipeline)
print("This is a  news ", predict_label , ":",ag_news_label[predict_label])
