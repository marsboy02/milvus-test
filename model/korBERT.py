from transformers import BertModel, BertTokenizer
import torch


class KorBERT:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KorBERT, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        model_name = "skt/kobert-base-v1"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def embed_sentence(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze().mean(dim=0).numpy()


if __name__ == "__main__":
    korBERT = KorBERT()
    sentence = "안녕하세요, 반갑습니다."
    embedding_vector = korBERT.embed_sentence(sentence)
    print(embedding_vector)
