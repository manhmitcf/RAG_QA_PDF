from langchain_community.document_loaders import PyPDFLoader, TextLoader


class PDFLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return PyPDFLoader(self.path).load()
    @property
    def get_len_of_doc(self):
        return len(self.load())
    

if __name__ == "__main__":
    loader = PDFLoader("data/2412.15308v1.pdf")
    print(loader.get_len_of_doc)
