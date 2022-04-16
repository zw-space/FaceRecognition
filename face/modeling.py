# -*- coding:utf-8 -*-
# -*- 收集人脸 -*-
# -*- 建立模型 -*-

import os
import cv2
import numpy as np
from PIL import Image
import tqdm
import json

# 配置文件
with open("config\\config.json", 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)['dataPath']
# 调用人脸分类器
face_detector = cv2.CascadeClassifier(config['face'])


def get_pic():
    # 为录入人脸标记
    face_id = input('User Name: ')
    if face_id in ['n', '']:
        return
    # 调用本地摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # 字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 图像保存地址
    img_path = "face_img\\" + face_id
    # 如不存在，则创建
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    # 该地址下文件个数（该文件一旦创建后请勿做任何修改
    img_count = len(os.listdir(img_path))
    # 设置样本最大值，数据越多，识别越好，但内存占用...
    # 如果感觉不准，可以适当增大
    count = 100
    while count > 0:
        # 从摄像头读取图片
        status, img = cap.read()
        if not status:
            break
        # 转为灰度图片，提高识别度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸，1.2为每次缩小比例，5为最小邻距
        face_point = face_detector.detectMultiScale(gray, 1.2, 5)
        # 显示
        cv2.imshow('Record Face', img)
        # 给人脸上框x, y, w, h分别代表左上角坐标和髋高
        for x, y, w, h in face_point:
            # 记录
            count -= 1
            # 根据左上角和右下角坐标画框，(0, 255, 0)为框的RGB值
            cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0))
            # 保存人脸图像，路径要确保存在，否则不会保存
            img_count += 1
            cv2.imwrite(os.path.join(img_path, str(img_count) + '.jpg'), gray[y:y + h, x:x + w])
            # 显示图片
            cv2.putText(img, str(count), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
            cv2.imshow('Record Face', img)
        # 取一帧，不要为0
        k = cv2.waitKey(1)
        # 按ESC退出/样本数据到1000后也会退出
        if k == 27:
            break

    # 关闭摄像头，释放资源
    cap.release()
    # 关闭窗口
    cv2.destroyAllWindows()


def train():
    if not os.path.exists('face_img'):
        raise Exception("你没有创建文件夹[face_img]")

    # 获取用于训练的图片路径
    users = [user for user in os.listdir("face_img")]
    # 存放用户图像的id文件
    with open('config\\user_id.json', 'r') as user_id_file:
        user_id = json.load(user_id_file)
    user_id['user'] = []
    user_id['id'] = []
    # 预定义图像id
    img_id = 0
    img_path_list = []
    # 遍历取出图像地址，保存图像user_id
    for up in users:
        for img_path in os.listdir('face_img\\' + up):
            img_id += 1
            img_path_list.append('face_img\\' + up + '\\' + img_path)
        else:
            user_id['user'].append(up)
            user_id['id'].append(img_id)

    # 保存用户和id
    with open('config\\user_id.json', 'w') as user_id_file:
        json.dump(user_id, user_id_file, indent=4)
    if len(img_path_list) == 0:
        raise Exception("你还没有任何数据")
    # 存放图像的数组
    face_images = []
    # 图像id
    img_id = 0
    img_ids = []
    # 遍历所有图像
    for img_path in tqdm.tqdm(img_path_list, desc="Training", leave=False):
        # 打开图片，并灰度图片
        img = Image.open(img_path).convert('L')
        # 图片转数组
        img = np.array(img, 'uint8')
        faces = face_detector.detectMultiScale(img)
        # 保存图像/id
        for (x, y, w, h) in faces:
            img_id += 1
            img_ids.append(img_id)
            face_images.append(img[y:y + h, x:x + w])

    # 初始化LBPH识别模型
    LBPH = cv2.face.LBPHFaceRecognizer_create()
    # 训练模型
    LBPH.train(face_images, np.array(img_ids))
    print("训练模型完成")
    # # 保存模型
    LBPH.save(config['train'])


if __name__ == '__main__':
    get_pic()
    train()
