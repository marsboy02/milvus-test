import os
import pandas as pd
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from model.embedder import TabularEmbedder

# 전역변수 설정
dimension = 128  # 벡터 차원 설정

# 환경변수 로드
load_dotenv()
database_url = os.getenv("DATABASE_URL")

# Milvus 연결
connections.connect("default", host=database_url, port="19530")

# Milvus 컬렉션 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    FieldSchema(name="columns", dtype=DataType.VARCHAR, max_length=65535)
]

schema = CollectionSchema(fields, description="Schema to store vector and other related data in Milvus")

# 기존 컬렉션 삭제 및 새로운 컬렉션 생성
collection_name = "hello_milvus"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
hello_milvus = Collection(collection_name, schema)

# 임베딩 처리
dataset_dir = "./dataset/"
embedder = TabularEmbedder(dataset_dir=dataset_dir)
embeddings_list = embedder.process_files()
print("Embedding process completed.")

# 데이터 준비 및 삽입
ids, titles, vectors, columns = [], [], [], []
for i, (title, embedding, column_names) in enumerate(embeddings_list):
    ids.append(i)
    titles.append(title)
    vectors.append(embedding)
    columns.append(" ".join(column_names))

entities = [ids, titles, vectors, columns]
insert_result = hello_milvus.insert(entities)
hello_milvus.flush()

# 인덱스 생성 및 로드
print("Creating index with FLAT configuration.")
index_params = {
    "index_type": "FLAT",
    "metric_type": "COSINE",
}
hello_milvus.create_index("vector", index_params)
hello_milvus.load()

# 검색 파라미터 설정
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}

# 검색 수행
query_titles = [title for title, _, _ in embeddings_list]
search_results = []
for i, vector in enumerate(vectors):
    result = hello_milvus.search([vector], "vector", search_params, limit=218, output_fields=["title"])
    query_title = query_titles[i]
    for hit in result[0]:
        search_results.append({
            "query_title": query_title,
            "target_title": hit.entity.get("title"),
            "distance": hit.distance,
            "id": hit.id,
        })

# 검색 결과 저장
df = pd.DataFrame(search_results)
df.to_csv("search_results.csv", index=False, encoding="utf-8-sig")
print("CSV file saved as 'search_results.csv'")
