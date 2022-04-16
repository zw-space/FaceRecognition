# -*- coding:utf-8 -*-
# -*- 识别 -*-

import cv2
import json

# 设置窗口可以自由拉伸/没啥用
# cv2.namedWindow('Recognizer', cv2.WINDOW_NORMAL)
# 调用本地摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)
# 字体
font = cv2.FONT_HERSHEY_SIMPLEX
# 配置文件
with open("config\\config.json", 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)['dataPath']
# 调用人脸分类器
face_detector = cv2.CascadeClassifier(config['face'])
# 初始化LBPH识别模型
LBPH = cv2.face.LBPHFaceRecognizer_create()
LBPH.read(config['train'])
# 用户对应的图像id文件
with open('config\\user_id.json', 'r') as user_id_file:
    user_id_dict = json.load(user_id_file)
    users = user_id_dict['user']
    user_ids = [0] + user_id_dict['id']

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 识别人脸
    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    # 进行校验
    if len(faces) == 0:
        cv2.imshow('Recognizer', img)
    for (x, y, w, h) in faces:
        flag = False
        # 进行检验，返回图片id和置信度（一般置信度高于80都不是一个好的结果）
        user_id, confidence = LBPH.predict(gray[y:y + h, x:x + w])
        if confidence < 40:
            color = (0, 255, 0)
            for i in range(len(users)):
                if user_ids[i] < user_id <= user_ids[i+1]:
                    user = users[i]
                    break
            else:
                user = "without you"
            confidence = "{0}%".format(round(100 - confidence))
        else:
            color = (0, 255, 255)
            user = "without you"
            confidence = "{0}%".format(round(100 - confidence))
        # 输出用户名和判断值
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(user), (x + 5, y - 5), font, 1, color, 1)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, color, 1)
        # 展示结果
        cv2.imshow('Recognizer', img)
    # 取一帧
    k = cv2.waitKey(1)
    # 按ESC退出
    if k == 27:
        break

# 释放资源
cap.release()
# 关闭窗口
cv2.destroyAllWindows()
