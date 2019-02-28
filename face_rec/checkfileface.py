'''
Created on 2019年2月13日

'''
import os,shutil
import numpy as np
# import dlib
import cv2
import sys
import traceback

class Config():
    face_features = []
    face_labels = []
#     detector = dlib.get_frontal_face_detector()
#     sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#     facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    check_min_distance=0.4
    # face_cascade = cv2.CascadeClassifier(r'cv2_xml/haarcascade_profileface.xml')
    face_cascade = cv2.CascadeClassifier(r'cv2_xml/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(r"cv2_xml/haarcascade_eye_tree_eyeglasses.xml")  
        
#将原图放在一个目录，以后只用原图
def getUseableFile(srcpath,targetpath):
    files = os.listdir(srcpath)
    files.sort(key= lambda x:int(x[:3]))
    for path in files:
        cursrcpath = srcpath+'/'+path
        curtargetpath = targetpath+'/'+path
        if os.path.isdir(cursrcpath):
            getUseableFile(cursrcpath, curtargetpath)
        else:
            try:
                if len(path) < 41 and hasface(cursrcpath):
                    dirname = os.path.dirname(curtargetpath)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    print(cursrcpath,curtargetpath)
                    shutil.copyfile(cursrcpath,curtargetpath)
            except (Exception):
                print("Unexpected error:",cursrcpath, sys.exc_info()[0],sys.exc_info()[1])
                
def hasface(imgpath,debug=False):
    img = cv2.imread(imgpath)  # 使用opencv读取图像数据
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2BGRA)
    
    # 探测图片中的人脸
    faces = Config.face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.25,
        minNeighbors = 1,
        minSize = (2,2),
        flags = cv2.FONT_HERSHEY_SIMPLEX
    )
    if debug:
        print(len(faces))
    if len(faces) > 0 and not debug :
        return True
        
    eyes = Config.eye_cascade.detectMultiScale(gray,1.1,2,cv2.CASCADE_SCALE_IMAGE,(2,2))  
    if debug:
        print(len(eyes))
    if len(eyes) > 0 and not debug:
        return True
    
    return False
                
def getfirstdescriptor(imgpath):
    img = cv2.imread(imgpath)  # 使用opencv读取图像数据
    print('getfirstdescriptor-imgpath:',imgpath)
    dets = Config.detector(img,2)
    if dets is None or len(dets) == 0:
        return None
    bb = np.zeros(4, dtype=np.int32)
    d = dets[0]
    bb[0] = np.maximum(d.left(), 0)
    bb[1] = np.maximum(d.top(), 0)
    bb[2] = np.minimum(d.right(), img.shape[1])
    bb[3] = np.minimum(d.bottom(), img.shape[0])
    rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
    shape = Config.sp(img, rec)
    face_descriptor = Config.facerec.compute_face_descriptor(img, shape)
    return face_descriptor
                
def initFace(labelimgpath='checkfileface_label'):
    face_labels = []
    face_features = []
    for path in os.listdir(labelimgpath):
        curpath = labelimgpath+'/'+path
        for imgpath1 in os.listdir(curpath):
            features = getfirstdescriptor(curpath+'/'+imgpath1)
            if features is not None:
                face_labels.append(path)
                face_features.append(features)
    return face_labels, np.array(face_features)
                
def find_most_likely_face(face_descriptor):
    face_distance = face_descriptor - Config.face_features
    euclidean_distance = 0
    if len(Config.face_labels) == 1:
        euclidean_distance = np.linalg.norm(face_distance)
    else:
        euclidean_distance = np.linalg.norm(face_distance, axis=1, keepdims=True)
    min_distance = euclidean_distance.min()
#     print('min_distance:',min_distance)
    if min_distance > Config.check_min_distance:
        return 'other'
    index = np.argmin(euclidean_distance)
    person_id = Config.face_labels[index]
    return person_id


def checkonepic(imgpath):
    img = cv2.imread(imgpath)  # 使用opencv读取图像数据
    dets = Config.detector(img,1)
    bb = np.zeros(4, dtype=np.int32)
    class_pres = []
    if dets is None or len(dets) == 0:
        print('no face!',imgpath)
        os.remove(imgpath)
    for k, d in enumerate(dets):
        bb[0] = np.maximum(d.left(), 0)
        bb[1] = np.maximum(d.top(), 0)
        bb[2] = np.minimum(d.right(), img.shape[1])
        bb[3] = np.minimum(d.bottom(), img.shape[0])
        rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        shape = Config.sp(img, rec)
        face_descriptor = Config.facerec.compute_face_descriptor(img, shape)
        class_pre = find_most_likely_face(face_descriptor)
#         print('class_pre:',class_pre)
        if not 'other' == class_pre:
            class_pres.append(class_pre)
    return class_pres

def checkpic(checkpath,resulttextpath='huhuhu.txt',resultpicpath='huhuhu'):
    files = os.listdir(checkpath)
    files.sort(key= lambda x:int(x[:3]))
    for path in files:
        curcheckpath = checkpath+'/'+path
        if os.path.isdir(curcheckpath):
            checkpic(curcheckpath,resulttextpath,resultpicpath)
        else:
            try:
                class_pres = checkonepic(curcheckpath)
                print(curcheckpath,str(len(class_pres)))
                if len(class_pres) > 0:
                    with open(resulttextpath, 'a') as f:
                        f.write(str(class_pres)+'\t'+curcheckpath+'\r\n')
                        #shutil.copyfile(curcheckpath,resultpicpath+'/')
            except (Exception):
                print('exception:',curcheckpath)
                traceback.print_exc()  
                #print("Unexpected error:",curcheckpath, sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2])
        
#将原图放在一个目录，以后只用原图
getUseableFile('G:/pub_pic','G:/pub_pic_clear_face')    

# hasface('G:/pub_pic_clear/003/001/011/00300101155_ed0c2c90.jpg')
# print(hasface('C:/Users/Administrator/Desktop/zawu/20190207/00300100920_4f3a685b.jpg',True))


# print('开始检查:',sys.argv[1],sys.argv[2],sys.argv[3])
# Config.face_labels,Config.face_features = initFace()
# checkpic(checkpath='pub_pic_clear_face')
#python checkfileface.py pub_pic_clear_face2 huhuhu2.txt huhuhu
# checkpic(checkpath=sys.argv[1],resulttextpath=sys.argv[2],resultpicpath=sys.argv[3])