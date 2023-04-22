import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from NeuralNetwork import bag_of_words, tokenize, stem
from brain import NeuralNet

with open("intents.json", 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = [',','?','/','.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

EPOCHS = 1000
BATCHSIZE = 8
LERNINGRATE = 0.001
INPUTSIZE = len(X_train[0])
HIDDENSIZE = 8
OUTPUTSIZE = len(tags)

print("Training The Model ......")


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.X_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset,
                          batch_size=BATCHSIZE,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(INPUTSIZE, HIDDENSIZE, OUTPUTSIZE).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LERNINGRATE)

for epoch in range(EPOCHS):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

print(f'Final Loss : {loss.item():.4f}')

data = {
    'model_state':model.state_dict(),
    'input_size':INPUTSIZE,
    'hidden_size': HIDDENSIZE,
    'output_size': OUTPUTSIZE,
    'all_words': all_words,
    'tags': tags
}

FILE = 'TrainData.pth'
torch.save(data, FILE)
print('Training Completed.')