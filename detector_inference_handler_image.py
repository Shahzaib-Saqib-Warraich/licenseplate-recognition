import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2

class Detector:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.device = torch.device('cpu') #cuda:0 for GPU OR cpu for CPU
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, force_reload=False)
        self.model.conf = 0.4
        self.model.to(self.device)
        print("YOLOv5 Model loaded successfully!")


    def detect(self, frame, size=(512, 512)):
        with torch.no_grad():
            prediction = self.model(frame, size=size)
            prediction = prediction.xyxy[0].cpu().numpy()

        return prediction


if __name__ == "__main__":

    img = cv2.imread("image1.jpg")
    cv2.imshow("input", img)
    cv2.waitKey(0)

    model_path="best.pt"

    yolov5 = Detector(model_path)

    output = yolov5.detect(img)

    for i in output:
        print(i)

        p1, p2 = (int(i[0]), int(i[1])), (int(i[2]), int(i[3]))
        cv2.rectangle(img, p1, p2, (0,0,255) , thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, 'plate', (p1[0], p1[1] - 2), 0, 1 / 3, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    
    cv2.imshow("test_output.jpg", img)
    cv2.waitKey(0)

    #Output = x1,y1,x2,y2,class_confidence, class