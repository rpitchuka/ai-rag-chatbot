import os
import fitz
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

def load_docs_from_uploads(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_content = extract_text_from_uploaded_file(file)
        doc = Document(page_content=file_content)
        documents.append(doc)
    return documents

def extract_text_from_uploaded_file(uploaded_file):
    file_in_bytes = uploaded_file.read()
    file_in_memory = fitz.open(stream=file_in_bytes, filetype="pdf")

    file_extracted_text = ""
    for page in file_in_memory:
        file_extracted_text += page.get_text("text") + "\n"
    return file_extracted_text
    

def split_loaded_docs(uploaded_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500, 
        chunk_overlap = 240, 
        separators=["\n\n", "\n", ".", ";", ":", ",", " "])
    split_docs = text_splitter.split_documents(uploaded_docs)
    return split_docs

def get_llm():
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.9, groq_api_key = os.getenv("GROQ_API_KEY"))
    return llm

def get_prompt():
    
    prompt = ChatPromptTemplate.from_template(
        """
        Use the context to answer the question. Be creative but do not make up answers.
        <context>
        {context}
        </context>

        Question: {input}
        """
    )
    return prompt

def build_retriever_chain(vectors, llm, prompt):
    retriver = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriver, document_chain)
    return retriever_chain

##Streamlit UI

st.title("ClauseWise - AI Powered Contract Analyzer")

with st.sidebar:
    st.header("Upload your Contracts — Let AI Do the Rest!")
    uploaded_files = st.file_uploader("Make sure your contracts are in PDF format before uploading.",accept_multiple_files=True, type=["pdf"])

    if st.button("Upload and Process"):
        with st.spinner("Processing uploaded contracts..."):
            uploaded_docs = load_docs_from_uploads(uploaded_files)
            split_docs = split_loaded_docs(uploaded_docs)
            embedder = OpenAIEmbeddings()
            st.session_state.vectors = FAISS.from_documents(split_docs, embedder)
        st.success("Files Uploaded and Processed")

user_query = st.text_input("Have contract-related queries? We’ve got you covered!")

if user_query and "vectors" in st.session_state:

    llm = get_llm()
    prompt = get_prompt()
    chain = build_retriever_chain(st.session_state.vectors, llm, prompt)
    response = chain.invoke({"input": user_query})
    st.write(response["answer"])
    
    ##display relevant context for the query
    with st.sidebar.expander("Source Snippets from Your Contracts"):
        for i, doc in enumerate(response['context']):
            snippet = doc.page_content[:500].replace("\n", " ")
            st.markdown(f"**Chunk {i+1}:** {snippet}...")