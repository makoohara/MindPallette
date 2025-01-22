import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration."""
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///db.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    TESTING = True


class TestingConfig(Config):
    """Testing configuration."""
    
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:' 