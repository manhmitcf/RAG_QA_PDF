from langchain_experimental.text_splitter import SemanticChunker 

class SemanticSplitter(SemanticChunker):
    def __init__(self, model_embedding):
        super().__init__(
            model_embedding=model_embedding,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            buffer_size=1,
            min_chunk_size=500,
            aad_start_index=True
        )
    
    def split_text(self, docs):
        return super().split_documnents(docs)