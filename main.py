import os
from pydoc import doc
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OllamaEmbeddings()
    docsearch = Pinecone.from_documents(
        texts,
        embeddings,
        index_name="medium-blogs-embeddings-index",
    )

    llm = ChatOllama(temperature=0, model="llama2")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    query = "What is a vector DB? Give me the answer in 15 words for a beginner."

    result = qa({"query": query})

    print(result)
