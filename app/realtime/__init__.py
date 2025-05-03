from flask import Blueprint

bp = Blueprint("realtime", __name__)

from app.realtime import routes
