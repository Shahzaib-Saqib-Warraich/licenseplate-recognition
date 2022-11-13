from easyocr import Reader
from collections import OrderedDict

class ocr():
    def __init__(self):
        self.labels_dict = OrderedDict()
        lang = ["en"]
        self.reader = Reader(lang, gpu=True)

    def predict(self,  crops_dict):
        for tr_id, crop in crops_dict.item():
            
            results = self.reader.readtext(crop[0])

            if results:
                for (bbox, text, prob) in results:
                    self.labels_dict[tr_id] = text
            else:
                continue

        return self.labels_dict
