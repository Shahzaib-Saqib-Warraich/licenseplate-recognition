from cv2 import mean
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import time
from PIL import Image
import pandas as pd

class Detector:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.device = torch.device('cpu') #cuda:0 for GPU OR cpu for CPU
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, force_reload=False)
        self.model.conf = 0.4  ## --------------------->> LICENSE PLATE DETECTOR THRESHOLD
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

    model_path="best.pt"
    yolov5 = Detector(model_path)

    ocr_result_list = []
    ocr_prob_list = []
    ocr_coordinates = []

    cap = cv2.VideoCapture('test_video_belta.mp4')
    # cap = cv2.VideoCapture(0)

    # ret,frame = cap.read()
    # h = frame.shape[0]
    # l = frame.shape[1]
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('output_video.mp4', fourcc, 5.0, (l,h))

    while(cap.isOpened()):
        ret,frame = cap.read()

        frame = cv2.rotate(frame, cv2.ROTATE_180) #Rotate


        if not ret:
            print('Frame not found')
            break
        
        t1 = time.time()

        detection_output = yolov5.detect(frame)

        for i in detection_output:
            # print(i)

            """
            i[0] = x1
            i[1] = y1
            i[2] = x2
            i[3] = y2
            """
            p1, p2 = (int(i[0]), int(i[1])), (int(i[2]), int(i[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255) , thickness=2, lineType=cv2.LINE_AA)
            
            crop_img = frame.copy()[int(i[1]): int(i[3]),  int(i[0]): int(i[2])]

            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
            # edged = cv2.Canny(bfilter, 30, 200) #Edge detection

            # img_name = str(t1)+'.jpg'
            # cv2.imwrite(img_name, bfilter)

            reader = easyocr.Reader(['en']) #language eng
            result = reader.readtext(bfilter)
            
            t2 = time.time()
            fps = round(1 / (t2-t1),3)
            # print("FPS: ", fps)

            if result:
                text = result[0][-2]
                ocr_prob = result[0][-1]
                if len(result)>1:
                    text0 = result[0][-2]
                    ocr_prob0 = result[0][-1]
                    text1 = result[1][-2]
                    ocr_prob1 = result[1][-1]
                    text = text0 + text1
                    ocr_prob = 0.5*(ocr_prob0+ocr_prob1)
                
                if ocr_prob>0.8: ## --------> OCR THRESHOLD
                    print('\nOCR Result:',text, '\tProbability',ocr_prob)
                    cv2.putText(frame, text , (p1[0], p1[1] - 5), 0, 2/3, (0, 50, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(frame, str(fps) , (10, 30), 0, 2 / 3, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                    ocr_result_list.append(text)
                    ocr_prob_list.append(ocr_prob)
                    # ocr_coordinates.append([p1,p2])


        cv2.imshow("test_output.jpg", frame)
        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        #Output = x1,y1,x2,y2,class_confidence, class

    # After the loop release the cap object
    cap.release()
    # out.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    # #Save into CSV
    # lst = [ocr_result_list,ocr_prob_list]
    # df = pd.DataFrame(lst, columns =['Plate Number', 'Probability'])
    # # saving the dataframe
    # df.to_csv('results_logging.csv')
    