# Create virtual environment
python -m venv ocr_env

# Activate environment
ocr_env\Scripts\activate

# Install required packages
pip install fastapi uvicorn python-multipart mistralai python-dotenv

pip freeze > requirements.txt