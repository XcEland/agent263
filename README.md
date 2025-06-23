# Create virtual environment
python -m venv ocr_env
python -m venv face_env
# Activate environment
cd ../..
ocr-env\Scripts\activate.bat

face_env\Scripts\activate
# Install required packages
pip install fastapi uvicorn python-multipart mistralai python-dotenv

pip freeze > requirements.txt

pip install deepface opencv-python numpy python-multipart aiofiles

uvicorn ocr_api:app --reload
ngrok http http://localhost:8000
ngrok config check
ngrok config edit
ngrok start --all