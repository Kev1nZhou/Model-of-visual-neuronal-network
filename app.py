import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

from models import load_snn_model, load_cnn_model, encode, decode
from models import End2EndTrain, CAE, DenseDecoder

import torchvision.transforms as transforms
import torchvision.datasets as datasets

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)


def main():
    # 加载模型

    snn_model = load_snn_model()
    cnn_model = load_cnn_model()

    # 创建窗口

    root = tk.Tk()
    root.title("演示程序")

    # 创建标签

    camera_text_label = Label(root, text="摄像头")
    camera_text_label.grid(row=0, column=0)
    sample_text_label = Label(root, text="截取图像")
    sample_text_label.grid(row=0, column=1)
    rebuild_text_label = Label(root, text="重建结果")
    rebuild_text_label.grid(row=0, column=2)

    # 创建图像框(以空白图像为初始化)

    blank_img = Image.new("RGB", (320, 320), (255, 255, 255))
    blank_imgtk = ImageTk.PhotoImage(image=blank_img)

    camera_label = Label(root, image=blank_imgtk)
    camera_label.grid(row=1, column=0)
    sample_label = Label(root, image=blank_imgtk)
    sample_label.grid(row=1, column=1)
    rebuild_label = Label(root, image=blank_imgtk)
    rebuild_label.grid(row=1, column=2)

    # 创建按钮

    capture_button = tk.Button(root, text="截取")
    capture_button.grid(row=2, column=1)
    process_button = tk.Button(root, text="重建")
    process_button.grid(row=2, column=2)

    # 初始化摄像头

    cap = cv2.VideoCapture(0)

    # 定义回调函数

    def update_camera_feed():
        """
        说明: 从摄像头获取并更新图像
        """
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.crop((80, 0, 560, 480)).resize((320, 320))
            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.config(image=imgtk)
            camera_label.image = imgtk

        # 每33毫秒刷新一次，模拟每秒30帧
        root.after(33, update_camera_feed)

    def capture_image():
        """
        说明: 从摄像头获取并降采样到合适尺寸
        """
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.crop((80, 0, 560, 480))
            img = img.resize((32, 32), Image.Resampling.BOX)
            imgLg = img.resize((320, 320), Image.Resampling.NEAREST)
            imgtk = ImageTk.PhotoImage(image=imgLg)
            sample_label.config(image=imgtk)
            sample_label.image = imgtk
            sample_label.image_data = np.array(img)

    def process_image():
        """
        说明: 调用 SNN-CNN-Model 进行图像重建
        """

        process_button.config(state=tk.DISABLED)

        def update(imgtk):
            rebuild_label.config(image=imgtk)
            rebuild_label.image = imgtk

        def process():
            image_data = sample_label.image_data
            image_data = decode(cnn_model, encode(snn_model, image_data))
            img = Image.fromarray(image_data)
            img = img.resize((320, 320), Image.Resampling.NEAREST)
            imgtk = ImageTk.PhotoImage(image=img)

            rebuild_label.after(0, lambda: update(imgtk))
            process_button.after(0, lambda: process_button.config(state=tk.NORMAL))

        threading.Thread(target=process).start()

    # 绑定回调并开始事件循环

    capture_button.config(command=capture_image)
    process_button.config(command=process_image)
    update_camera_feed()
    root.mainloop()

    # 释放资源

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

