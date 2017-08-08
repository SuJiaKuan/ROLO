import cv2
import numpy as np
import sys
from networks.YOLO_network import YOLO_TF
from networks.ROLO_network import ROLO_TF

num_steps = 3  # number of frames as an input sequence
batch_size = 1

class TrackState:
    BEFORE_DRAW = 1
    DRAWING = 2
    TRACKING = 3

    def  __init__(self, state = BEFORE_DRAW):
        self.state = state

    def to(self, state):
        self.state = state

    def equal(self, state):
        return self.state == state

def main(argv):
    if len(argv) < 2:
        print "Usage: python {} source".format(argv[0])
        sys.exit(-1)

    source = argv[1]

    yolo = YOLO_TF()
    rolo = ROLO_TF()

    look_back_size = num_steps * batch_size
    previous_yolo_outputs = []
    pred_location = []
    state = TrackState(TrackState.DRAWING)
    idx = int(0)

    cap = cv2.VideoCapture(source)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if state.equal(TrackState.DRAWING):
            if idx == look_back_size:
                state.to(TrackState.TRACKING)
                pred_location = [1000, 160, 250, 800]
            else:
                idx += 1

        if state.equal(TrackState.TRACKING):
            yolo_output = yolo.compute_yolo_output(frame, pred_location)
            previous_yolo_outputs.append(yolo_output[0])
            if len(previous_yolo_outputs) > look_back_size:
                previous_yolo_outputs.pop(0)
            if len(previous_yolo_outputs) == look_back_size:
                pred_location = rolo.track(previous_yolo_outputs, frame)
                x = int(pred_location[0])
                y = int(pred_location[1])
                w = int(pred_location[2])
                h = int(pred_location[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('ROLO', frame)
                cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
