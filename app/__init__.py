from flask import Flask
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app 