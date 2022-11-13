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

    img = cv2.imread("original_6_result.bmp")
    cv2.imshow("test_input.jpg", img)
    cv2.waitKey(0)

    model_path="best.pt"

    # yolov5 = Detector(model_path)

    # output = yolov5.detect(img)

    # for i in output:
    #     print(i)

    """
    i[0] = x1
    i[1] = y1
    i[2] = x2
    i[3] = y2
    """
    # p1, p2 = (int(i[0]), int(i[1])), (int(i[2]), int(i[3]))
    # cv2.rectangle(img, p1, p2, (0,0,255) , thickness=2, lineType=cv2.LINE_AA)
        
    # crop_img = img.copy()[int(i[1]): int(i[3]),  int(i[0]): int(i[2])]
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    # img = cv2.Canny(bfilter, 30, 200) #Edge detection

    cv2.imshow("cropped", img)
    cv2.waitKey(0)

    # img_name = str(time.time())+'.jpg'
    # cv2.imwrite(img_name, bfilter)

    reader = easyocr.Reader(['en']) #language eng
    result = reader.readtext(img)
    
    if result:
        #
        text = result[0][-2]
        ocr_prob = result[0][-1]
        if len(result)>1:
            text0 = result[0][-2]
            ocr_prob0 = result[0][-1]
            text1 = result[1][-2]
            ocr_prob1 = result[1][-1]
            text = text0 + text1
            ocr_prob = 0.5*(ocr_prob0+ocr_prob1)
        #      
        # if ocr_prob>0.8: ## --------> OCR THRESHOLD
        print('\nOCR Result:',text, '\tProbability',ocr_prob)
         
    

    cv2.imshow("test_output.jpg", img)
    cv2.waitKey(0)

    #Output = x1,y1,x2,y2,class_confidence, class