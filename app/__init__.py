from flask import Flask, render_template
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Blueprint'leri kaydet
    from app.api import bp as api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    # Ana sayfa rotaları
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/contact")
    def contact():
        return render_template("contact.html")

    return app
