import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-1234'
    UPLOAD_FOLDER = 'app/static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size 