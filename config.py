#encoding: utf-8
import os

DEBUG = False

CAN_UPLOAD = True

SECRET_KEY = os.urandom(24)

SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:1234qwer@10.0.251.50/face'
SQLALCHEMY_TRACK_MODIFICATIONS = True


