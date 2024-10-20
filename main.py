from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv


load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search(
    "What is an interesting fact about the English language?",
    k=1 # returns a single result, the most relevant.
)

for result in results:
    print("\n")
    print(result.page_content)




"""
to print separates page_content

for doc in docs:
    print(doc.page_content)
    print("\n")
"""

"""
to print the score 

results = db.similarity_search_with_score

+

print(result[1])
    print(result[0].page_content)
"""

# Split the text into separate chuncks
# Calculate embeddings for each chunk
# Store embeddings in a db specialized in store embeddings
# Take user question and create an embedding out of it
# Do a similarity search with our stored embeddings to find the ones most similar to the user's question
# Put the most relevant 1-3 facts into de prompt along with the user's question
# pip install chromadb