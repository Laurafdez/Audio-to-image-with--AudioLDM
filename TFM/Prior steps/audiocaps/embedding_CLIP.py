import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)

import clip

clip.available_models()



model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)



import pandas as pd

df = pd.read_csv("train.csv")
captions = df["caption"]

embeddings = []
caption_embeddings = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for caption in captions:
    text_tokens = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float().to(device)
    embeddings.append(text_features.cpu().numpy())
    caption_embeddings.append(caption)

embedding_df = pd.DataFrame({
    "caption": caption_embeddings,
    "embedding": embeddings
})
embedding_df.to_csv("CLIP.csv", index=False)


