import torch
from torchtext.legacy import data
import re
import spacy
import random
import torchtext.vocab as vocab
import torch.nn as nn
import torch.optim as optim

SEED = 2022

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

spacy_ger = spacy.load("de_core_news_sm")


def cleanup_text(texts):
    cleaned_text = []
    for text in texts:
        # remove punctuation
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        # remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # remove newline
        text = re.sub(r'\n', ' ', text)

        cleaned_text.append(text)
    return cleaned_text


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


TEXT = data.Field(tokenize=tokenize_ger,
                  preprocessing=cleanup_text,
                  batch_first=True,
                  include_lengths=True,
                  lower=True)

LABEL = data.LabelField(dtype=torch.float)

fields = [(None, None), ('text', TEXT), (None, None), ('label', LABEL)]

training_data = data.TabularDataset(path='application/dataset_service.tsv',
                                    fields=fields,
                                    format='tsv',
                                    skip_header=True)

train_data, valid_data = training_data.split(split_ratio=0.7, random_state=random.seed(SEED))

custom_embeddings = vocab.Vectors(name='application/vectors.txt')

TEXT.build_vocab(train_data, min_freq=5, vectors=custom_embeddings, max_size=15279)
LABEL.build_vocab(train_data)

# No. of unique tokens in text
print("Size of TEXT vocabulary:", len(TEXT.vocab))

# No. of unique tokens in label
print("Size of LABEL vocabulary:", len(LABEL.vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)


class Classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs


size_of_vocab = len(TEXT.vocab)
embedding_dim = 300
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

model = Classifier(size_of_vocab,
                   embedding_dim,
                   num_hidden_nodes,
                   num_output_nodes,
                   num_layers,
                   bidirectional=True,
                   dropout=dropout)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

model = model.to(device)
criterion = criterion.to(device)

model.load_state_dict(torch.load('application/model.pt'))
model.eval()


def predict(m, sentences):
    predicts = []
    for sentence in sentences:
        tokenized = [tok.text for tok in spacy_ger.tokenizer(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1).T
        length_tensor = torch.LongTensor(length)
        prediction = m(tensor, length_tensor)
        if 0 <= prediction.item() <= 0.33:
            predicts.append('neutral')
        elif 0.33 < prediction.item() <= 0.66:
            predicts.append('positive')
        elif 0.66 < prediction.item() <= 1:
            predicts.append('negative')
    return predicts
