print("🔄 Initializing local RAG pipeline...")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("🔄 Processing your document locally...")

# 1. Load the text file
loader = TextLoader("knowledge.txt")
docs = loader.load()

# 2. Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Vectorize text completely locally using Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. Define the prompt template
template = """You are a helpful assistant. Answer the question based ONLY on the following context:
{context}

Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 5. Initialize the local LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0)

# 6. Format documents helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Construct the LangChain Pipeline using the pipe (|) operator
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Run a test query
print("\n🤖 AI is ready. Asking question...")
query = "hello?"
response = rag_chain.invoke(query)

print(f"\nQuestion: {query}")
print(f"Response: {response}")
