from flask_login import UserMixin
from . import db


class User(UserMixin, db.Model):
    """User model for authentication and user management."""
    
    __tablename__ = 'user'
    __table_args__ = {'extend_existing': True}
    
    id = db.Column(
        db.Integer, 
        primary_key=True
    )  # Primary key required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))


class History(db.Model):
    """History model for storing user diary entries and generated content."""
    
    __tablename__ = 'history'
    
    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime, nullable=False)
    diary_entry = db.Column(db.String(500), nullable=False)
    generated_image = db.Column(db.String(500), nullable=False)
    song_snippet = db.Column(db.String(500), nullable=False)
    user_id = db.Column(
        db.Integer, 
        db.ForeignKey('user.id'), 
        nullable=False
    )