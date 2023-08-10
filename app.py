"""
author @ kumar dahal
"""
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy import *
import pymysql
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import (
    UserMixin,
    login_user,
    LoginManager,
    login_required,
    current_user,
    logout_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_toastr import Toastr


pymysql.install_as_MySQLdb()
rdsConnection = create_engine(
    "mysql+mysqldb://admin:passwordRDS@visionrds.ch0vw6wkyqa3.ca-central-1.rds.amazonaws.com/"
)

# creates the database instance (SQLAlchemy)
db = SQLAlchemy()
DB_NAME = "users"
app = Flask(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
toastr = Toastr(app)

RDS_ENDPOINT = 3306
USERNAME = "admin"
PASSWORD = "passwordRDS"
DB_NAME = "users"

app.config["SECRET_KEY"] = "test"
app.config[
    "SQLALCHEMY_DATABASE_URI"
] = f"mysql://admin:passwordRDS@visionrds.ch0vw6wkyqa3.ca-central-1.rds.amazonaws.com/vision"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    name = db.Column(db.String(150))


with app.app_context():
    db.create_all()


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        return render_template("index.html", user=current_user)
    except Exception as e:
        return str(e)


# Route to handle sign-in page


@app.route("/register", methods=["POST", "GET"])
def register():
    print("test")
    if request.method == "POST":
        email = request.form.get("email")
        name = request.form.get("name")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if user:
            flash("Email already exists.", category="error")
        else:
            print("test1")
            if len(email) < 4:
                flash("Email must be greater than 4 characters.", category="error")
            elif len(name) < 2:
                flash("Name must be greater than 2 characters.", category="error")
            elif len(password) < 6:
                flash("Password must be greater than 6 characters.", category="error")
            else:
                new_user = User(
                    email=email,
                    name=name,
                    password=generate_password_hash(password, method="sha256"),
                )
                print(new_user)
                db.session.add(new_user)
                db.session.commit()
                flash("Account created successfully.", category="success")
                return redirect(url_for("register"))
        return render_template("register.html", user=current_user)
    else:
        return render_template("register.html", user=current_user)


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        email = request.form.get("email")  # getting the user email from the login form
        password = request.form.get(
            "password"
        )  # getting the user password from the login form

        user = User.query.filter_by(
            email=email
        ).first()  # checking if the user email is in the database
        if user:
            if check_password_hash(
                user.password, password
            ):  # checking in the database if the user password is correct
                flash(
                    "Logged in successfully!", category="success"
                )  # to show the success messagesss
                login_user(user, remember=True)
                return redirect(
                    url_for("index", user=current_user)
                )  # redirecting to user page if the login is successful
            else:
                flash(
                    "Incorrect password, try again.", category="error"
                )  # user password is incorrect
        else:
            flash(
                "Email does not exist.", category="error"
            )  # if the email does not exist in the database
    return render_template("login.html", user=current_user)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


login_manager.init_app(app)

# @app.route('/user', methods=['GET'])
# def user():
#     return render_template("user.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'alert-danger')
    # toastr.info("This is a success message", title="Success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=8001)
