from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Load configuration settings
    app.config.from_object('config')
    
    # Import and register blueprints
    from .routes import main
    app.register_blueprint(main)
    
    return app