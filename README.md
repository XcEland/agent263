# Create virtual environment
python -m venv ocr_env
python -m venv face_env
# Activate environment
cd ..
ocr-env\Scripts\activate.bat

# Install required packages
pip install -r requirements.txt
pip freeze > requirements.txt

# Run api
cd automation-agents
uvicorn ocr_api:app --reload
To access the api docs: http://localhost:8000/docs

# Run with ngrok
pip install 
ngrok http http://localhost:8000
ngrok config check
ngrok config edit
ngrok start --all