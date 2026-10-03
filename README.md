# 1. Create a virtual environment named 'ai_env'
python3 -m venv ai_env

# 2. Activate the virtual environment
source ai_env/bin/activate

# 3. Install the LangChain packages safely inside this environment
pip install langchain langchain-openai langchain-community faiss-cpu


create data file: knowledge.txt, paste some custom facts into it that an AI model wouldn't know by default

create the Main Script (app.py)


python app.py        //cost required...

use Ollama
