Create a virtual environment named 'ai_env'

$ python3 -m venv ai_env

Activate the virtual environment

$ source ai_env/bin/activate

Install the LangChain packages safely inside this environment

$ pip install langchain langchain-openai langchain-community faiss-cpu


create data file: knowledge.txt, paste some custom facts into it that an AI model wouldn't know by default


create the Main Script (app.py)


python app.py        //cost required...



use Ollama:

$ curl -fsSL https://ollama.com/install.sh | sh

$ ollama run llama3.2:1b

then you can send messages now, quit: ctrl + d

$ ollama pull mxbai-embed-large

$ source ai_env/bin/activate

$ pip install langchain-ollama

update app.py

$ python app.py    //you can ask questions by replacing the query in app.py

//app.py

8. Run a test query

print("\n🤖 AI is ready. Asking question...")

query = "hello?"

response = rag_chain.invoke(query)


