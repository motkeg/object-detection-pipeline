import cv2
import time
from src.bb2geo.streams_intersect import StreamIntersectCalculator

DREW_CONTOUR_ON_MAP = True
MAP_PATH = "/hdd/musashi/data/warehouse_setup_27_08_19_1920X1080.jpg"
# MAP_PATH = "src/runners/map_with_contours.jpg"
COLORS = [(255, 0, 0)]


def draw(detection_data, matrix=None):
    print("Start Drawing...")
    if matrix is not None:
        intersection_calculator = StreamIntersectCalculator(MAP_PATH,[matrix])
    while True:
        val = detection_data.get()
        for key, dets in val.items():

            frame, cnts = dets
            if int(key) % 50 == 0:
                print("draw frame ", key)

            if DREW_CONTOUR_ON_MAP:
                assert matrix is not None
                intersection_calculator.draw_on_map(cnts=cnts, colors=COLORS, view_map=False,
                                                    cnts_threshold_interval=(700,10000))

            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < 600 or cv2.contourArea(c) > 100000:
                    # if cv2.contourArea(c) > 500:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, str(time.ctime(time.time())) + " f:{}".format(key), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # show the frame
            cv2.imshow("FrameDetector", frame)
            intersection_calculator.save_map_image()
            if cv2.waitKey(100) & 0xFF == 27:
                break
