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


