from byte_tracker import BYTETracker

class ByteTracker_yolov5():
    def __init__(self):
      
        self.track_thresh = 0.15
        self.track_buffer = 0.9
        self.match_thresh = 0.9
        self.min_box_area = 1
        self.mot20 = False

        self.byte_tracker   = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh, self.min_box_area, self.mot20)
        self.in_img_size    = (512, 512)
        self.out_img_size   = (512, 512)

    def to_tlbr(self, tlwh):
        tlbr = np.empty(4)
        xmin = float(tlwh[0])
        ymin = float(tlwh[1])
        tlbr[0] = int(round(xmin, 0))
        tlbr[1] = int(round(ymin, 0))
        tlbr[2] = int(round(xmin + float(tlwh[2]) - 1., 0))
        tlbr[3] = int(round(ymin + float(tlwh[3]) - 1., 0))
        return tlbr

    def process_detections(self, detections):
        processed_detections = []
        for detection in detections:
            tlwh = detection[:4]
            class_id= detection[4]
            score = detection[5]

            tlbr = self.to_tlbr(tlwh)
            tlbr = np.append(tlbr, score)
            tlbr = np.append(tlbr, class_id)
            processed_detections.append(tlbr)

        processed_detections = np.array(processed_detections)
        return processed_detections

    def track(self, detections):
        processed_detections = self.process_detections(detections=detections)
        if len(processed_detections):
            track_bbs_ids = self.byte_tracker.update(processed_detections, self.in_img_size, self.out_img_size)
        else:
            track_bbs_ids=[]

        return track_bbs_ids
       
    def apply(self, detections):
        tracked_people  = self.track(detections)