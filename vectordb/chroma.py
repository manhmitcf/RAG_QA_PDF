from langchain_chroma import Chroma 
from langchain_core.vectorstores import VectorStoreRetriever
class ChromaVectorStore(Chroma):
    def __init__(
        self,
        embedding_function = None,
        persist_directory: str = "./chroma_db",
        collection_name: str = "default"
    ):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function

        super().__init__(
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_name=collection_name)
    def add_documents(self, docs):
        super().add_documents(docs)
        self.persist()

    def query(self, query_text: str, k: int = 3):
        return self.similarity_search(query_text, k=k)

    def query_with_score(self, query_text: str, k: int = 3):
        return self.similarity_search_with_score(query_text, k=k)
    def as_retriever(self, search_kwargs: dict = None):
        return VectorStoreRetriever(vectorstore=self, search_kwargs=search_kwargs or {"k": 3})
    def delete_all(self):
        self._collection.delete(where={})
        self.persist()
        print("Tất cả dữ liệu đã bị xoá khỏi Chroma.")


if __name__ == "__main__":
    chroma = ChromaVectorStore()


    