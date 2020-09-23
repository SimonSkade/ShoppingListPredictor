from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
import os


app = Flask(__name__)#the heart of the app (or actually the whole app)

#configuration
#distinguish between server deploy (ENV=production) and local testing
if os.environ.get("ENV") == "production":
	app.config["DEBUG"] = False
	app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")#get secret key from env
	app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")#get database from environment
	train_on = "cpu"
else:
	app.config["DEBUG"] = True
	app.config["SECRET_KEY"] = "7g2fab9567unr8jvndush9zjn2hbnoug5ut"#just set this secret key
	app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"#use or deploy a local database
	app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # set the default max file age to 0
	train_on = "cuda"

db = SQLAlchemy(app)#this is the database
bcrypt = Bcrypt(app)#stuff for password encryption
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"


from webapp import routes


