import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re
from torch import tensor
import re
class EmbeddingDataset(Dataset):
    def __init__(self, path_files):
        self.path_files = path_files
        self.files = os.listdir(self.path_files)

    def read_embeddings(self, archivo_ruta):
        with open(archivo_ruta, 'r') as archivo:
            lineas = archivo.readlines()

        clip_embeddings = []
        clap_embeddings = []
        leer_clip = False
        leer_clap = False

        for linea in lineas:
            if linea.startswith('CLIP Embedding:'):
                leer_clip = True
                leer_clap = False
            elif linea.startswith('CLAP Embedding:'):
                leer_clip = False
                leer_clap = True
            elif leer_clip and linea.strip():
                clip_embeddings.append(np.fromstring(linea, sep=' ').astype(np.float32))
            elif leer_clap and linea.strip():
                clap_embeddings.append(np.fromstring(linea, sep=' '))


        clip_embeddings = np.array(clip_embeddings)
        clap_embeddings = np.array(clap_embeddings)
        

        clip_embeddings = torch.tensor(clip_embeddings, dtype=torch.float32)
        clap_embeddings = torch.tensor(clap_embeddings, dtype=torch.float32)
    
        
        return clip_embeddings, clap_embeddings

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip, clap = self.read_embeddings(self.path_files + self.files[idx])

        return clip, clap


path_directory_train = "/home/byo/Amazon/laura/dataset/buenos/clip/train/"
path_directory_test = "/home/byo/Amazon/laura/dataset/buenos/clip/test/"

training_data = EmbeddingDataset(path_directory_train)
test_data = EmbeddingDataset(path_directory_test)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


# Defining the neural network architecture
class CLIptoCLAP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CLIptoCLAP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = torch.relu(self.fc1(x.reshape(x.size(0), -1)))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.normalize(out, p=2, dim=1)
        return out


# Define the hyperparameters
input_dim = 512  
hidden_dim = 256  
output_dim = 512  
learning_rate = 0.001
num_epochs = 50

# Model
model = CLIptoCLAP(input_dim, hidden_dim, output_dim)

#Defining the loss function and the optimiser
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training of the model
train_losses = []
test_losses = []


for epoch in range(num_epochs):
    # Poner el modelo en modo de entrenamiento
    model.train()
    train_loss = 0

 
    for clip_embedding, target_clap_embedding in train_dataloader:
        
        predicted_clap_embedding = model(clip_embedding)
        predicted_clap_embedding = predicted_clap_embedding.unsqueeze(2)
        loss = criterion(predicted_clap_embedding, target_clap_embedding)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    
    average_train_loss = train_loss / len(train_dataloader)
    train_losses.append(average_train_loss)

    
    model.eval()
    test_loss = 0

    for clip_embedding, target_clap_embedding in test_dataloader:
        with torch.no_grad():
            predicted_clap_embedding = model(clip_embedding)
            predicted_clap_embedding = predicted_clap_embedding.unsqueeze(2)
            loss = criterion(predicted_clap_embedding, target_clap_embedding)
            test_loss += loss.item()

    
    average_test_loss = test_loss / len(test_dataloader)
    test_losses.append(average_test_loss)

    
    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, average_train_loss, average_test_loss))


plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Calcular el promedio de los errores de entrenamiento y prueba
average_train_loss = sum(train_losses) / len(train_losses)
average_test_loss = sum(test_losses) / len(test_losses)

# Imprimir el promedio de los errores
print('Promedio de errores de entrenamiento: {:.4f}'.format(average_train_loss))
print('Promedio de errores de prueba: {:.4f}'.format(average_test_loss))
model.eval()
test_loss = 0

for clip_embedding, target_clap_embedding in test_dataloader:
    with torch.no_grad():
        predicted_clap_embedding = model(clip_embedding)
        predicted_clap_embedding = predicted_clap_embedding.unsqueeze(2)
        loss = criterion(predicted_clap_embedding, target_clap_embedding)
        test_loss += loss.item()

average_test_loss = test_loss / len(test_dataloader)
print('Average Test Loss: {:.4f}'.format(average_test_loss))


torch.save(model.state_dict(), 'modelEMC256.pth')
