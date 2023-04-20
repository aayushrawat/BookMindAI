from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
import pinecone
import streamlit as st
import os

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def pdf_to_data(file_here):
    loader = UnstructuredPDFLoader(file_here)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0)
    text = text_splitter.split_documents(data)
    return text

# Initialize Pinecone


pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
index_name = "bookimind-ai"


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")


st.title("BookMindAI")
st.write("Upload your PDF here")
with st.form("my_form"):
    uploaded_file = st.file_uploader("Choose a file")
    st.write("Write your query here")
    query = st.text_input('Query')
    submitted = st.form_submit_button("Submit")
    if submitted:
        texts = pdf_to_data(uploaded_file)
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
        docs = docsearch.similarity_search(query, include_metadata=True)
        result = chain.run(input_documents=docs, question=query)
        st.write(result)


