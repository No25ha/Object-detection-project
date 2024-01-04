import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.root.geometry("800x600")

        self.canvas = tk.Canvas(self.root, width=600, height=400)
        self.canvas.pack(pady=20)

        self.btn_open = tk.Button(self.root, text="Open Image", command=self.open_image)
        self.btn_open.pack(pady=10)

        self.btn_detect = tk.Button(self.root, text="Detect Objects", command=self.detect_objects)
        self.btn_detect.pack(pady=10)

        self.image = None
        self.image_cv2 = None
        self.image_path = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image = image.resize((600, 400), Image.ANTIALIAS)
            self.image = ImageTk.PhotoImage(image)
            self.image_cv2 = cv2.imread(file_path)
            self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)

    def apply_gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    def detect_objects(self):
        if self.image_cv2 is not None:
            self.image_cv2 = self.apply_gaussian_blur(self.image_cv2)

            net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
            blob = cv2.dnn.blobFromImage(self.image_cv2, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    class_id = int(detections[0, 0, i, 1])
                    x1, y1, x2, y2 = (
                        detections[0, 0, i, 3:7] * [self.image_cv2.shape[1], self.image_cv2.shape[0],
                                                     self.image_cv2.shape[1], self.image_cv2.shape[0]]).astype("int")

                    cv2.rectangle(self.image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.update_canvas()

    def update_canvas(self):
        image = cv2.cvtColor(self.image_cv2, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 400), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()
