import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


class TransformerEmbedder(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2):
        super(TransformerEmbedder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1,
        )
        self.fc = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add a sequence dimension
        x = self.transformer(x, x)
        x = x.mean(dim=1)  # Pool over the sequence dimension
        x = self.fc(x)
        return x


class TabularEmbedder:
    def __init__(self, dataset_dir="./dataset/", model_dim=128):
        self.dataset_dir = dataset_dir
        self.model_dim = model_dim

    def preprocess_data(self, df):
        df.fillna(0, inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    def embed(self, df):
        input_dim = df.shape[1]
        model = TransformerEmbedder(input_dim, model_dim=self.model_dim)
        tensor = torch.tensor(df.values, dtype=torch.float32)
        embedding = model(tensor)
        return embedding.detach().numpy()

    def process_files(self):
        csv_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.csv')]
        embeddings_list = []

        for csv_file in csv_files:
            print(f"Processing file: {csv_file}")  # CSV 파일 이름 출력
            file_path = os.path.join(self.dataset_dir, csv_file)
            df = pd.read_csv(file_path)
            df = self.preprocess_data(df)

            embedding = self.embed(df)
            column_names = df.columns.tolist()

            embeddings_list.append((embedding.mean(axis=0), column_names))

        return embeddings_list
