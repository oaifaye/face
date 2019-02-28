#encoding: utf-8

from flask import Flask, render_template, request, redirect, url_for, session
import config
from models import *
from exts import db
from decorators import login_required
from sqlalchemy import or_
import os
from face_rec.get_face_feature_vec import FaceFeatures
import numpy as np
import cv2
import sys
from PIL import ImageDraw, Image, ImageFont

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)

def initFace(qiangzhi=False):
    if len(FaceFeatures().face_labels) == 0 or qiangzhi:
        face_features = []
        face_labels = []
        face_name_dict = {}
        faces = TfFace.query.all()
        for face in faces:
            face_features.append(np.fromstring(face.vec,sep=' '))
            face_labels.append(face.person_id)
        FaceFeatures.face_features = np.array(face_features) 
        FaceFeatures.face_labels = face_labels
        persons = TfPerson.query.all()
        for person in persons:
            face_name_dict[person.id] = person.person_name
        FaceFeatures.face_name_dict = face_name_dict
        print('FaceFeatures.face_features',FaceFeatures().face_features)
        print('FaceFeatures.face_labels',FaceFeatures().face_labels)
        print('FaceFeatures.face_name_dict',FaceFeatures().face_name_dict)
        

@app.route('/')
def index():
    initFace()
    context = {
        'questions': Question.query.all()
    }
    return render_template('index.html', **context)

@app.route('/personlist/')
def personlist():
    initFace()
    person = TfPerson.query.order_by('-init_date').all()
    canupload = config.CAN_UPLOAD
    context = {
        'person': person,
        'size' : len(person),
        'canupload' : config.CAN_UPLOAD
    }
    return render_template('person_list.html', **context)

@app.route('/addperson/', methods=['GET', 'POST'])
def addperson():
    initFace()
    if request.method == 'GET':
        person = TfPerson(person_name='', state=1)
        return render_template('person.html',person=person)
    else:
        person_name = request.form.get('person_name')
        person = TfPerson(person_name=person_name, state=1)
        jump = request.form.get('jump')
        db.session.add(person)
        db.session.commit()
        if jump == '1':
            return redirect('/uploadpage/'+str(person.id))
        return redirect(url_for('personlist'))
    
@app.route('/updateperson/<person_id>', methods=['GET', 'POST'])
def updateperson(person_id):
    initFace()
    if request.method == 'GET':
        person = TfPerson.query.filter(TfPerson.id == person_id).first()
        return render_template('person.html',person=person)
    else:
        person_name = request.form.get('person_name')
        id = request.form.get('id')
        jump = request.form.get('jump')
        person = TfPerson.query.filter(TfPerson.id == id).first()
        person.person_name = person_name #将查询出来的所有的title =='aaa1'的记录，全部修改为'aaa111'
        db.session.commit()
        if jump == '1':
            return redirect(url_for('index'))
        return redirect(url_for('personlist'))
  
@app.route('/uploadpage/<person_id>', methods=['GET'])
def uploadpage(person_id):
    initFace()
    person = TfPerson.query.filter(TfPerson.id == person_id).first()
    faces = TfFace.query.filter(TfFace.person_id == person_id)
    return render_template('upload.html', person=person,faces=faces)

@app.route('/upload', methods=['post'])
def upload():
    initFace()
    img = request.files.get('file')
    name = request.form.get("name")
    person_id = request.form.get("person_id")
    path = 'static/faces/'+person_id+"/"
    if not os.path.exists(path):
        os.makedirs(path) 
    img_path = path+name
    img.save(img_path)
    
    #获取图片对特征向量
    faceFeatures = FaceFeatures()
    face_features = faceFeatures.get_face_features(img_path)
    if face_features is None:
        return ''
    face = TfFace(person_id=person_id, vec=str(face_features[0][0]).replace('[','').replace(']',''),file_name=name)
    db.session.add(face)
    db.session.commit()
    return ''

@app.route('/predict', methods=['post'])
def predict():
#     try:
        img = request.files.get('file')
        name = request.form.get("name")
        img_path = 'static/faces_predict/'+name
        pointindex = img_path.rindex('.')
        img_path_h = img_path[0:pointindex] + '_h' + img_path[pointindex:len(img_path)]
        img.save(img_path)
        
        img = cv2.imread(img_path)  # 使用opencv读取图像数据
        print('img.shape[0] * img.shape[1]:',str(img.shape[0] * img.shape[1]))
        if img.shape[0] * img.shape[1] > 400000:  # 对大图可以进行压缩，阈值可以自己设置
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        FaceFeatures().predict(img,img_path_h)
        return img_path_h
#     except (Exception):
#         print("Unexpected error:", sys.exc_info()[0],sys.exc_info()[1])
#         return ''

@app.route('/reflashface', methods=['post'])
def reflashface():
    initFace(qiangzhi=True)
    return ''
