import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


class TabTransformer(nn.Module):
    def __init__(self, num_categories, num_continuous_features, embed_dim=128, num_heads=4, num_layers=2):
        super(TabTransformer, self).__init__()
        # 범주형 변수를 위한 임베딩 레이어 생성
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, embed_dim) for cat_size in num_categories])
        
        # 연속형 데이터를 임베딩 크기에 맞게 변환하는 선형 레이어 (입력 크기를 num_continuous_features로 설정)
        self.continuous_embed = nn.Linear(num_continuous_features, embed_dim)
        
        # Transformer 블록 생성
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers)
        
        # 최종 출력 레이어
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_cat, x_cont):
        # 범주형 데이터 임베딩
        x_cat_emb = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.size(1))]
        x_cat_emb = torch.stack(x_cat_emb, dim=1)
        
        # 연속형 데이터가 있는 경우, 임베딩 차원으로 변환
        if x_cont is not None:
            x_cont_emb = self.continuous_embed(x_cont)
            x_cat_emb += x_cont_emb.unsqueeze(1)  # 차원을 맞추기 위해 확장 후 더하기
        
        # Transformer 적용
        x_transformed = self.transformer(x_cat_emb)
        x_out = self.fc(x_transformed.mean(dim=1))  # Pooling 후 최종 출력
        return x_out


class TabularEmbedder:
    def __init__(self, dataset_dir="./dataset/", embed_dim=128):
        self.dataset_dir = dataset_dir
        self.embed_dim = embed_dim

    def preprocess_data(self, df):
        df.fillna(0, inplace=True)  # 결측값을 0으로 채움
        
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = df.select_dtypes(exclude=['object']).columns.tolist()
        
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        return df, categorical_columns, numerical_columns, label_encoders

    def embed(self, df, categorical_columns, numerical_columns):
        num_categories = [df[col].nunique() for col in categorical_columns]
        num_continuous_features = len(numerical_columns)  # 연속형 피처의 개수를 계산
        x_cat = torch.tensor(df[categorical_columns].values, dtype=torch.long)
        x_cont = torch.tensor(df[numerical_columns].values, dtype=torch.float32) if numerical_columns else None
        
        model = TabTransformer(num_categories, num_continuous_features, embed_dim=self.embed_dim)
        embedding = model(x_cat, x_cont)
        return embedding.detach().numpy()

    def process_files(self):
        csv_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.csv')]
        embeddings_list = []

        for csv_file in csv_files:
            print(f"Processing file: {csv_file}")
            file_path = os.path.join(self.dataset_dir, csv_file)
            df = pd.read_csv(file_path)
            df, categorical_columns, numerical_columns, label_encoders = self.preprocess_data(df)

            embedding = self.embed(df, categorical_columns, numerical_columns)
            column_names = df.columns.tolist()

            embeddings_list.append((csv_file, embedding.mean(axis=0), column_names))

        return embeddings_list
