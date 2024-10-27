import pandas as pd
from model.embedder import TabularEmbedder
import os
from dotenv import load_dotenv
import json
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# 전역변수 선언
dimension = 128  # 벡터 차원을 실제 사용되는 차원 수로 설정합니다.

# env 설정
load_dotenv()
database_url = os.getenv("DATABASE_URL")
connections.connect("default", host=database_url, port="19530")

# 밀버스에 적용할 스키마 작성
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    FieldSchema(name="columns", dtype=DataType.VARCHAR, max_length=65535)
]

schema = CollectionSchema(fields, "Schema to store vector and other related data in Milvus")
if utility.has_collection("hello_milvus"):
    utility.drop_collection("hello_milvus")
hello_milvus = Collection("hello_milvus", schema)

# embedding
dataset_dir = "./dataset/"
embedder = TabularEmbedder(dataset_dir=dataset_dir)
embeddings_list = embedder.process_files()
print("embedding is ok")

# insert
ids, titles, vectors, columns = [], [], [], []
for i, (title, embedding, column_names) in enumerate(embeddings_list):
    ids.append(i)
    titles.append(title)
    vectors.append(embedding)
    columns.append(" ".join(column_names))
entities = [ids, titles, vectors, columns]

insert_result = hello_milvus.insert(entities)
hello_milvus.flush()

# index
print(format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}
hello_milvus.create_index("vector", index)
hello_milvus.load()


search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}
result = hello_milvus.search(vectors, "vector", search_params, limit=194, output_fields=["title"])


search_results = []
for hits in result:
    for hit in hits:
        search_results.append({
            "id": hit.id,
            "distance": hit.distance,
            "title": hit.entity.get("title")
        })

df = pd.DataFrame(search_results)
df.to_csv("search_results.csv", index=False, encoding="utf-8-sig")
print("CSV file saved as 'search_results.csv'")
