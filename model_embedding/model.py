
from langchain_huggingface import HuggingFaceEmbeddings
import torch
class ModelEmbedding:
    def __init__(self, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {'device': device}
        self.model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    def get_embedding_context(self, text):
        return self.model.embed_documents(text)

    def get_embedding_query(self, text):
        return self.model.embed_query(text)
    

if __name__ == "__main__":
    model = ModelEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    print(model.get_embedding_context("Hello world"))