#encoding: utf-8
import config
from flask import Flask, render_template, request, redirect, url_for, session
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from platform import app, initFace
from exts import db
from models import User, Question, Answer
from face_rec.get_face_feature_vec import FaceFeatures
import dlib

manager = Manager(app)

# 使用Migrate绑定app和db
migrate = Migrate(app, db)

# 添加迁移脚本的命令到manager中
manager.add_command('db', MigrateCommand)

FaceFeatures.detector = dlib.get_frontal_face_detector()
FaceFeatures.sp = dlib.shape_predictor('face_rec/shape_predictor_68_face_landmarks.dat')
FaceFeatures.facerec = dlib.face_recognition_model_v1('face_rec/dlib_face_recognition_resnet_model_v1.dat')
# FaceFeatures.face_features,FaceFeatures.face_labels = initFace()
# print('FaceFeatures.face_features',FaceFeatures().face_features)
# print('FaceFeatures.face_labels',FaceFeatures().face_labels)

if __name__ == '__main__':
    manager.run()