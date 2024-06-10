from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


def create_db_from_path(
    input_path: str, embeddings_model: str, output_path: str, split: bool = False
) -> None:
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(
        input_path,
        glob="*.py",
        recursive=True,
        loader_cls=TextLoader,
        loader_kwargs=text_loader_kwargs,
    )
    documents = loader.load()
    if split:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=embeddings_model)
    db = FAISS.from_documents(documents, embeddings)

    db.save_local(output_path)
