import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from PIL import Image
import time

class Detector:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.device = torch.device('cpu') #cuda:0 for GPU OR cpu for CPU
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, force_reload=False)
        self.model.conf = 0.4
        self.model.to(self.device)
        # self.transform = transforms.Compose([
        #     transforms.Resize((512, 512), interpolation=3),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        print("YOLOv5 Model loaded successfully!")


    def detect(self, frame, size=(512, 512)):
        with torch.no_grad():
            prediction = self.model(frame, size=size)
            prediction = prediction.xyxy[0].cpu().numpy()

        return prediction


if __name__ == "__main__":

    img = cv2.imread("image2.jpg")
    cv2.imshow("test_input.jpg", img)
    cv2.waitKey(0)

    model_path="best.pt"

    yolov5 = Detector(model_path)

    output = yolov5.detect(img)

    for i in output:
        print(i)

        """
        i[0] = x1
        i[1] = y1
        i[2] = x2
        i[3] = y2
        """
        p1, p2 = (int(i[0]), int(i[1])), (int(i[2]), int(i[3]))
        cv2.rectangle(img, p1, p2, (0,0,255) , thickness=2, lineType=cv2.LINE_AA)
        
        crop_img = img.copy()[int(i[1]): int(i[3]),  int(i[0]): int(i[2])]
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
        # edged = cv2.Canny(bfilter, 30, 200) #Edge detection
    
        # cv2.imshow("cropped", bfilter)
        # cv2.waitKey(0)

        # img_name = str(time.time())+'.jpg'
        # cv2.imwrite(img_name, bfilter)

        reader = easyocr.Reader(['en']) #language eng
        result = reader.readtext(bfilter)
        if result:
            text = result[0][-2]
            ocr_prob = result[0][-1]
            print('\nOCR Result:',text, '\tProbability',ocr_prob)
            cv2.putText(img, text , (p1[0], p1[1] - 5), 0, 2 / 3, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow("test_output.jpg", img)
    cv2.waitKey(0)

    #Output = x1,y1,x2,y2,class_confidence, class