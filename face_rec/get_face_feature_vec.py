import config
from flask import Flask, render_template, request, redirect, url_for, session
import dlib
import numpy as np
import cv2
import os
import json
import argparse
from exts import db

import sys
from models import *
from PIL import ImageDraw, Image, ImageFont

class FaceFeatures():
    
             
    face_features=[]
    face_labels=[]
    face_name_dict = {}
    
    def find_most_likely_face(self,face_descriptor):
        face_distance = face_descriptor - self.face_features
        euclidean_distance = 0
        if len(self.face_labels) == 1:
            euclidean_distance = np.linalg.norm(face_distance)
        else:
            euclidean_distance = np.linalg.norm(face_distance, axis=1, keepdims=True)
        min_distance = euclidean_distance.min()
        print('min_distance:',min_distance)
        if min_distance > 0.4:
            return 'other'
        index = np.argmin(euclidean_distance)
        person_id = self.face_labels[index]
        if self.face_name_dict[person_id] is None:
            return person_id
        
        return self.face_name_dict[person_id]
                
    def predict(self,img,img_path_h):
        dets = self.detector(img,2)
        bb = np.zeros(4, dtype=np.int32)
        recs = []
        class_pres = []
        for k, d in enumerate(dets):
            bb[0] = np.maximum(d.left(), 0)
            bb[1] = np.maximum(d.top(), 0)
            bb[2] = np.minimum(d.right(), img.shape[1])
            bb[3] = np.minimum(d.bottom(), img.shape[0])
            rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
            shape = self.sp(img, rec)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape)
            class_pre = self.find_most_likely_face(face_descriptor)
            recs.append(rec)
            print('class_pre:',class_pre)
            class_pres.append(class_pre)
        
        #画框
        for rec,class_pre in zip(recs, class_pres):
            cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0,0, 255), 2)
        
        #写字
        # cv2转PIL
        img  = Image.fromarray(img) 
        for rec,class_pre in zip(recs, class_pres):
            img = self.myPutText(img, str(class_pre), rec.left()-30, rec.top()-30, 50,20)
        #PIL图片转cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2BGRA)#COLOR_RGB2BGR
        cv2.imwrite(img_path_h, img)   
        
    '''在图片上打印中文'''
    def myPutText(self,pil_im,perName,x,y,w,h):
        # cv2转PIL
#         pil_im  = Image.fromarray(image)  
        #PIL图片上打印中文
        draw = ImageDraw.Draw(pil_im) # 括号中为需要打印的canvas，这里就是在图片上直接打印
        font = ImageFont.truetype("simhei.ttf", 16, encoding="utf-8") # 第一个参数为字体文件路径，第二个为字体大小
        draw.text((int((x+x+w)/2),int((y+y+h)/2)), perName, (0,0 , 255), font=font) # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
        #PIL图片转cv2
#         cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_BGR2BGRA)#COLOR_RGB2BGR
        return pil_im 
        
    def get_face_features(self,path_img):
        img = cv2.imread(path_img)
        if img.shape[0] * img.shape[1] > 400000:  # 对大图可以进行压缩，阈值可以自己设置
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        
        dets = self.detector(img, 2)  # 使用检测算子检测人脸，返回的是所有的检测到的人脸区域
        if len(dets) == 0:
            print("没有检测的人脸图像：", path_img, "\n")
            return None
        print("检测的人脸图像：", path_img, "\n")
        d = dets[0]     # 默认处理第一个检测到的人脸区域
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(d.left(), 0)
        bb[1] = np.maximum(d.top(), 0)
        bb[2] = np.minimum(d.right(), img.shape[1])
        bb[3] = np.minimum(d.bottom(), img.shape[0])
    
        rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        
        shape = self.sp(img, rec)  # 获取landmark
        
        face_descriptor = self.facerec.compute_face_descriptor(img, shape)  # 使用resNet获取128维的人脸特征向量
        face_array = np.array(face_descriptor).reshape((1, 128))  # 转换成numpy中的数据结构
    
        return face_array,bb

    