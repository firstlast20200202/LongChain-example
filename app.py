import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

print("🔄 Processing your document...")

# 2. Load the text file
loader = TextLoader("knowledge.txt")
docs = loader.load()

# 3. Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 4. Vectorize text and store it in a fast local database (FAISS)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. Define the prompt template
template = """You are a helpful assistant. Answer the question based ONLY on the following context:
{context}

Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 7. Format documents helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. Construct the LangChain Pipeline using the pipe (|) operator
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Run a test query
print("\n🤖 AI is ready. Asking question...")
query = "Who is leading Project Nebula and what is the latency goal?"
response = rag_chain.invoke(query)

print(f"\nQuestion: {query}")
print(f"Response: {response}")
