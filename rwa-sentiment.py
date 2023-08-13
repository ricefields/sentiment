
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.indexes import VectorstoreIndexCreator

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
#import unstructured
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from langchain import PromptTemplate
from langchain import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain import ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings


from langchain.chains.question_answering import load_qa_chain

import streamlit as st
import pandas as pd
import numpy as np

os.environ["OPENAI_API_KEY"] = "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"

#x=load_dotenv()
#print(x)

#llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = OpenAI(model_name="gpt-4", temperature=0)

st.title ("Sentiment Analysis")
#st.header ("RWA")
#st.subheader("Sentiment Analysis")
#st.markdown ("I am markdown")
#st.code ("I am a code block. E.g.: int i = 10")
#st.text ("I am text")
#st.write ("I am write")

#df = pd.DataFrame(
#    np.random.randn(50,20),
#    columns = ('col %d' % i for i in range(20)))

#st.write (df)

#Download text to user
#sample_text = "some text"
#st.download_button ('download text', sample_text)

message = st.text_input ('Intented Message', 'All is Well')

template = """
You are required to perform sentiment analysis on supplied text.
Please provide the estimated polarity scores of the following text {message}."
"""

prompt = PromptTemplate(template=template, input_variables=["message"])
#st.write ("I am processing your input. Please wait until the 'Start Sentiment Analysis' button appears in steady red")
llm_chain = LLMChain(prompt=prompt, llm=llm)
    
response = llm_chain.run(message)
print (response)

if (st.button("Start Sentiment Analysis")):
    st.write (response)