from model.embedder import TabularEmbedder


def main():
    dataset_dir = "./dataset/"
    embedder = TabularEmbedder(dataset_dir=dataset_dir)
    embeddings_list = embedder.process_files()

    for embedding, column_names in embeddings_list:
        print(f"Embedding: {embedding}")
        print(f"Column Names: {column_names}")


if __name__ == "__main__":
    main()
