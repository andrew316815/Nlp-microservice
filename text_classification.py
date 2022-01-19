# -*- coding: utf-8 -*-
"""text_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LlxsokFn2uDvkbLaE9I1rHZWvv-LszIY

# Sentiment analysis

## Импорт библиотек
"""

import torch   
from torchtext.legacy import data 
import pandas as pd

"""## Просмотр датасета"""

df = pd.read_csv('dataset_service.tsv', sep='\t', comment='#', header=None)
df.head()

SEED = 2022

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

"""# Предобработка датасета"""

# !python -m spacy download de

import re
import spacy  

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

"""## Определение колонок необходимых для модели"""

fields = [(None, None), ('text', TEXT), (None, None), ('label', LABEL)]

training_data = data.TabularDataset(path = 'dataset_service.tsv', fields = fields, format = 'tsv', skip_header = True)

#print preprocessed text
print(vars(training_data.examples[0]))

"""## Разделение данных на тренировочные и тестовые"""

import random
train_data, valid_data = training_data.split(split_ratio=0.7, 
                                             random_state = random.seed(SEED))

"""## Загрузка germany embeddings """

# Download model
# !wget https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt

import torchtext.vocab as vocab

custom_embeddings = vocab.Vectors(name = 'vectors.txt')

TEXT.build_vocab(train_data, min_freq=5, vectors=custom_embeddings, max_size=15279)
LABEL.build_vocab(train_data)

#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

#Commonly used words
print(TEXT.vocab.freqs.most_common(10))  

#Word dictionary
print(TEXT.vocab.stoi)   

#Label dictionary
print(LABEL.vocab.stoi)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

BATCH_SIZE = 64

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True,
    device = device)

"""## Классификатор"""

import torch.nn as nn

class classifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):

        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        #activation function
        self.act = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]
      
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)
        
        return outputs

"""## Гиперпараметры"""

size_of_vocab = len(TEXT.vocab)
embedding_dim = 300
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

#instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)

"""## Информация о модели"""

print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

"""## Вспомогательные функции"""

import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

 
def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
    
model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
     
    model.train()  
    i = 0
    for batch in iterator:
        i += 1
        print(f'batch #{i}')
        #resets the gradients after every batch
        optimizer.zero_grad()   
        
        #retrieve text and no. of words
        text, text_lengths = batch.text   
        
        #convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()  
        
        #compute the loss
        loss = criterion(predictions, batch.label)        
        
        #compute the binary accuracy
        acc = binary_accuracy(predictions, batch.label)   
        
        #backpropage the loss and compute the gradients
        loss.backward()       
        
        #update the weights
        optimizer.step()      
        
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #deactivating dropout layers
    model.eval()
    
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            #retrieve text and no. of words
            text, text_lengths = batch.text
            
            #convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()
            
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

"""## Обучение модели"""

# N_EPOCHS = 3
# best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):
#    print(f'epoch {epoch}')
#    #train the model
#   train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    
    #evaluate the model
#    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    #save the best model
#    if valid_loss < best_valid_loss:
#       best_valid_loss = valid_loss
#        torch.save(model.state_dict(), 'model.pt')
    
#    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

"""## Загрузка модели"""

#load weights
model.load_state_dict(torch.load('model.pt'));
model.eval();

def predict(model, sentences):
    predicts = []
    for sentence in sentences:
        tokenized = [tok.text for tok in spacy_ger.tokenizer(sentence)]    
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]           
        length = [len(indexed)]                                     
        tensor = torch.LongTensor(indexed).to(device)              
        tensor = tensor.unsqueeze(1).T                              
        length_tensor = torch.LongTensor(length)                   
        prediction = model(tensor, length_tensor)
        if 0 <= prediction.item() <= 0.33:
            predicts.append('neutral')
        elif 0.33 < prediction.item() <= 0.66:
            predicts.append('positive')
        elif 0.66 < prediction.item() <= 1:
            predicts.append('negative')              
    return predicts

"""## Предсказание"""

print('result: ', predict(model, ['ich hasse diese Welt', 'Ich habe gute Laune' ] ))