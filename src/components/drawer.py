import cv2
import time
import numpy as np
from src.bb2geo.streams_intersect import StreamIntersectCalculator ,StreamTransformer
from src.bb2geo.utils import create_colors_list

DREW_CONTOUR_ON_MAP = True
MAP_PATH = "/hdd/musashi/data/warehouse_setup_27_08_19_1024X768.jpg"
# MAP_PATH = "src/runners/map_with_contours.jpg"
COLORS = [(255, 0, 0)]
INTERVAL = (500,10000)

def draw(detection_data, matrix=None):
    print("Start Drawing...")
    if matrix is not None:
        intersection_calculator = StreamIntersectCalculator(MAP_PATH,"/hdd/musashi/output/contoures/cam1",
                                                            StreamTransformer(matrix),StreamTransformer(matrix))

    # fourcc = cv2.VideoWriter_fourcc(*'AVI')
    # video_writer = cv2.VideoWriter('contour_map_vs_frame_output.avi', fourcc, 20.0, (640, 480))
    while True:
        val = detection_data.get()
        for key, dets in val.items():

            frame, cnts = dets

            if int(key) % 50 == 0:
                print("draw frame ", key)

            # filter out the small and big contour
            cnts = list(filter(lambda c: INTERVAL[0] <= cv2.contourArea(c) <= INTERVAL[1], cnts))
            colors = create_colors_list(len(cnts))
            if DREW_CONTOUR_ON_MAP:
                assert matrix is not None
                contour_map = intersection_calculator.draw_on_map(cnts=cnts, colors=colors, view_map=False)

            for i,c in enumerate(cnts):
                # if the contour is too small, ignore it

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                # rect = cv2.minAreaRect(c)
                # box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(frame, [c], 0, colors[i], 2)

            contour_frame = cv2.putText(frame, str(time.ctime(time.time())) + " f:{}".format(key), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # stack the map and contours
            if contour_map is not None:
                contour_stack = np.hstack((contour_map, cv2.resize(contour_frame,(contour_map.shape[1],contour_map.shape[0]))))

            else:
                contour_stack = contour_frame
            # show the frame
            cv2.imshow("FrameDetector", contour_stack)
            if int(key) % 10 == 0:
                cv2.imwrite(f'/hdd/musashi/output/contoures/cam1/frame_{key}.jpg',contour_stack)
            # video_writer.write(contour_stack)
            # intersection_calculator.save_map_image()
            if cv2.waitKey(20) & 0xFF == 27:
                break
