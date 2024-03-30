from flask import Blueprint, render_template, redirect, url_for, request, flash
# from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
from .models import User
from . import db
from flask_bcrypt import Bcrypt

auth = Blueprint('auth', __name__)
bcrypt = Bcrypt()


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.query.filter_by(email=email).first()

        if not user or not bcrypt.check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('auth.login'))

        login_user(user, remember=remember)
        print("LOGGEEEDDDD", current_user)
        return redirect(url_for('main.home'))
    
    if current_user.is_authenticated:
        return redirect(url_for('main.home', user=current_user))  # Assuming 'main.index' is the name of the function that renders your home page
    return render_template('login.html', user={'is_authenticated': False})


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('main.home', user=current_user))

    print("methodddd aaaaa:", request.method)
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user:
            flash('Email address already exists')
            return redirect(url_for('auth.signup'), user={'is_authenticated': False})

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(email=email, name=name, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('auth.login'), user={'is_authenticated': False})

    return render_template('signup.html', user={'is_authenticated': False})


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.home', user={'is_authenticated': False}))

