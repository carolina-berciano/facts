from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

"""
This file will run anytime we want to ask some question of ChatGPT 
and use some content out of our vector databse to provide some context 
"""

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA._chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff" # take some context from the vector store eand stuff it into the prompt
)                      # SystemMessagePromptTemplate, HumanMessagePromptTemplate

result = chain.run("What is an interesting fact about the English language?")
print(result)
