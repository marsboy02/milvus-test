from model.embedder import TabularEmbedder
import random
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

connections.connect("default", host="localhost", port="19530")


# Define the fields for the schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="columns", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
]

# Create the collection schema
schema = CollectionSchema(fields, "Schema to store vector and other related data in Milvus")

# Check if the collection exists and drop it if it does
if utility.has_collection("hello_milvus"):
    utility.drop_collection("hello_milvus")

# Create the collection
hello_milvus = Collection("hello_milvus", schema)


dataset_dir = "./dataset/"
embedder = TabularEmbedder(dataset_dir=dataset_dir)
embeddings_list = embedder.process_files()

print("embedding is ok")

# Initialize lists for each field
ids = []
columns = []
vectors = []

for i, (embedding, column_names) in enumerate(embeddings_list):
    ids.append(i)
    columns.append(json.dumps(column_names))  # JSON 문자열로 변환
    vectors.append(embedding)

entities = [
    ids,
    columns,
    vectors
]

print("entity is ok")
print(entities)

insert_result = hello_milvus.insert(entities)
hello_milvus.flush()

print(format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("vector", index)

hello_milvus.load()
vectors_to_search = vectors[-1:]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
result = hello_milvus.search(vectors_to_search, "vector", search_params, limit=10, output_fields=["columns"])
print(result)
