from flask import url_for, render_template, flash, redirect, request, make_response
from webapp import app, db, bcrypt, login_manager
from flask_login import login_user, current_user, logout_user, login_required
from webapp.modules import User
from webapp.forms import RegistrationForm, LoginForm


@app.route("/")
@app.route("/home")
def home():
	return render_template("base.html")


@app.route("/about")
def about():
	return render_template("about.html")


@app.route("/register", methods=["GET","POST"])
def register():
	if current_user.is_authenticated:
		return redirect(url_for("home"))
	form = RegistrationForm()
	if form.validate_on_submit():
		hashed_password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
		user = User(username=form.username.data, email=form.email.data, password=hashed_password)
		db.session.add(user)
		db.session.commit()
		flash(f"Account created for {form.username.data}! You were logged in!", "success")
		login_user(user)
		next_page = request.args.get("next")
		return redirect(next_page) if next_page else redirect(url_for("home"))
	return render_template("register.html", form=form)


@app.route("/login", methods=["GET","POST"])
def login():
	if current_user.is_authenticated:
		return redirect(url_for("home"))
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		if user and bcrypt.check_password_hash(user.password, form.password.data):
			login_user(user)
			next_page = request.args.get("next")
			return redirect(next_page) if next_page else redirect(url_for("home"))
		else:
			flash("Login unsuccessful. Please check username and password", "danger")
	return render_template("login.html", form=form)

@app.route("/logout")
def logout():
	logout_user()
	return redirect(url_for("home"))

