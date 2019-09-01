import cv2
import time


def draw(detection_data):
    print("Start Drawing...")
    while True:
        val = detection_data.get()
        for key, dets in val.items():

            frame, cnts = dets
            if int(key) % 50 == 0:
                print("draw frame ", key)
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

            if cv2.waitKey(10) & 0xFF == 27:
                break
