import torch
import torch.nn as nn
from torchvision import transforms
import clip
from PIL import Image
import requests
from io import BytesIO
import os
import torch.nn.functional as F


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

# Cargar el modelo personalizado desde el archivo 'model.pth'
model_path = os.path.join(os.path.dirname(__file__), 'modelCL512mixdropaprend2.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model = CLIptoCLAP(512,512, 512)
custom_model.load_state_dict(torch.load(model_path))
custom_model = custom_model.to(device)
custom_model.eval()

def get_image_embedding(image_path):
    try:
        imagen = Image.open(image_path)
        if imagen is None:
            print("Error al abrir la imagen")
            return None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar el modelo preentrenado de CLIP
        model, preprocess = clip.load("ViT-B/32")
        model = model.to(device)
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        imagen = preprocess(imagen).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embed_clip = model.encode_image(imagen)
            embed_clip = embed_clip.to(device)
            embed_clip = embed_clip.to(custom_model.fc1.weight.dtype)
            embed_clip = embed_clip.view(embed_clip.size(0), -1)  # Redimensionar a dos dimensiones
            # Obtener los embeddings de CLAP utilizando el modelo personalizado
            embed_clap = custom_model(embed_clip)
            embed_clap = embed_clap.view(embed_clap.size(0), -1, embed_clap.size(1))  # Expandir segunda dimensión

        return embed_clap.detach()
    
    except Exception as e:
        print("Error al procesar la imagen:", str(e))
        return None


