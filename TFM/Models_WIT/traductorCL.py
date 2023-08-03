import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import tensor
import re
from torch.utils.data import ConcatDataset

class EmbeddingDataset(Dataset):
    def __init__(self, path_files):
        self.path_files = path_files
        self.files = os.listdir(self.path_files)

class EmbeddingDataset(Dataset):
    def __init__(self, path_files):
        self.path_files = path_files
        self.files = os.listdir(self.path_files)

    def read_embeddings(self, archivo_ruta):
        with open(archivo_ruta, 'r') as archivo:
            lineas = archivo.readlines()

        clip_embeddings_text = ""
        clap_embeddings_text = ""
        leer_clip = False
        leer_clap = False

        for linea in lineas:
            if linea.startswith('embedding CLIP'):
                leer_clip = True
                leer_clap = False
            elif linea.startswith('embedding CLAP'):
                leer_clip = False
                leer_clap = True
            elif leer_clip and linea.strip():
                clip_embeddings_text += linea
            elif leer_clap and linea.strip():
                clap_embeddings_text += linea

        # Eliminar el texto "embedding CLIP" y convertir a tensor
        clip_embeddings_text = clip_embeddings_text.replace("embedding CLIP", "").strip()

        clip_embeddings = torch.tensor(eval(clip_embeddings_text), dtype=torch.float32).to('cuda:0')

        # Eliminar el texto "embedding CLAP" y convertir a tensor
        clap_embeddings_text = clap_embeddings_text.replace("embedding CLAP", "").strip()
        # Supongamos que la variable tiene el siguiente valor
        

        # Utilizamos una expresión regular para eliminar la parte de grad_fn
        clap_embeddings_text = re.sub(r", device=.*?(?=,|\))", "", clap_embeddings_text)
        indice_paréntesis =  clap_embeddings_text.find(']')

        # Extraer la parte del tensor antes del primer paréntesis de cierre
        clap_embeddings_text = clap_embeddings_text[:indice_paréntesis + 1] 
        clap_embeddings_text = clap_embeddings_text.strip() + ")"
        clap_embeddings = torch.tensor(eval(clap_embeddings_text), dtype=torch.float32).to('cuda:0')

        return clip_embeddings, clap_embeddings

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip, clap = self.read_embeddings(self.path_files + self.files[idx])

        return clip, clap



path_directory_train = "/home/byo/Amazon/laura/database/datos_filtrados/train/"
path_directory_test = "/home/byo/Amazon/laura/database/datos_filtrados/test/"

training_data = EmbeddingDataset(path_directory_train)
test_data = EmbeddingDataset(path_directory_test)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True)



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


class SimCLR(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(SimCLR,self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        self.negatives_mask = self.negatives_mask.to('cuda:0')
        emb_i = emb_i.to('cuda:0')
        emb_j = emb_j.to('cuda:0')
        emb_j = torch.squeeze(emb_j, -1)
        

        z_i = torch.nn.functional.normalize(emb_i, dim=1)
        z_j = torch.nn.functional.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        
        similarity_matrix = torch.nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        
        #self.negatives_mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool).to(similarity_matrix.device)
        #self.register_buffer("negatives_mask", (~torch.eye(similarity_matrix.size(0), similarity_matrix.size(0), dtype=bool)).float())
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss




optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training of the model
train_losses = []
test_losses = []
num =0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    model = model.to('cuda:0')
    
    for clip_embedding, target_clap_embedding in train_dataloader:
        clip_embedding = clip_embedding.to('cuda:0')
        target_clap_embedding = target_clap_embedding.to('cuda:0')

        if clip_embedding.shape == torch.Size([64, 512,1]) and target_clap_embedding.shape == torch.Size([64, 512, 1]):
            predicted_clap_embedding = model(clip_embedding)
            
           
            loss_function_cl = SimCLR(batch_size=64)
            loss = loss_function_cl(predicted_clap_embedding, target_clap_embedding )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        else:
            num=num+1
            
    print(num)        
    average_train_loss = train_loss / len(train_dataloader)
    train_losses.append(average_train_loss)

    model.eval()
    test_loss = 0

    for clip_embedding, target_clap_embedding in test_dataloader:
        with torch.no_grad():
            if clip_embedding.shape == torch.Size([64, 512, 1]) and target_clap_embedding.shape == torch.Size([64, 512, 1]):
                predicted_clap_embedding = model(clip_embedding)
                loss_function_cl = SimCLR(batch_size=64)
                loss = loss_function_cl(predicted_clap_embedding, target_clap_embedding )


                test_loss += loss.item()

    average_test_loss = test_loss / len(test_dataloader)
    test_losses.append(average_test_loss)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, num_epochs, average_train_loss,
                                                                         average_test_loss))

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

average_train_loss = sum(train_losses) / len(train_losses)
average_test_loss = sum(test_losses) / len(test_losses)

print('Average Train Loss: {:.4f}'.format(average_train_loss))
print('Average Test Loss: {:.4f}'.format(average_test_loss))

