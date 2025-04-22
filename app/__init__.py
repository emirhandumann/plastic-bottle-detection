from flask import Flask, render_template
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    @app.route('/')
    def index():
        return render_template('index.html')

    return app 