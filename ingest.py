from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
#from langchain.vectorstores import Qdrant

from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders.image import UnstructuredImageLoader

class FileProcessor:
    def __init__(self, fileLocation):
        self.fileLocation = fileLocation

    def process(self,contentType):
        #matching the file types for loaders
        if contentType == "text/plain":
            loader = TextLoader(self.fileLocation)
            document = loader.load()
        elif contentType == "application/pdf":
            loader = PyPDFLoader(self.fileLocation)
            document = loader.load_and_split()
        elif contentType == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(self.fileLocation)
            document = loader.load()
        elif contentType == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            loader= UnstructuredExcelLoader(self.fileLocation)
            document = loader.load()
        elif contentType == "text/csv":
            loader = CSVLoader(self.fileLocation)
            document = loader.load()
        elif contentType == "application/vnd.ms-powerpoint" or contentType == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            loader = UnstructuredPowerPointLoader(self.fileLocation)
            document = loader.load()
        elif contentType == "image/png" or contentType == "image/jpeg" or contentType == "image/jpg":
            loader = UnstructuredImageLoader(self.fileLocation)
            document = loader.load()
        else:
            #for unsupported file type
            return []
            

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=600,
            chunk_overlap=300,
            length_function=len,
        )
        chunks = text_splitter.split_documents(document)
        # chunks = text_splitter.create_documents(document)
        return chunks
        # index = self.__indexDocument(chunks)

        # return index


    # def __indexDocument(chunks):
    #     # storing embeddings in the vector store
    #     # vectorstore = FAISS.from_texts(texts=chunks,
    #     #               embedding= embeddings)
    #     embeddings = select_embedding_model()
    #     vectorstore = Qdrant.from_documents(
    #         chunks,
    #         embeddings,
    #         location=":memory:",  # Local mode with in-memory storage only
    #         collection_name="my_documents",
    #     )
    #     return vectorstore
