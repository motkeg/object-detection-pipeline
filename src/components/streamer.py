import cv2
import numpy as np

FIRST_FRAME_IDX = 170
NUM_OF_FRAMES_TO_STACK = 50

COUNTER = 0
first_frames = list()


def read(frames_dict, path=0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FIRST_FRAME_IDX)
    first_frame, i = None, 0
    print("Start reading video...")
    global COUNTER
    while True:
        # Capture frame-by-frame
        while not frames_dict.full():
            ret, frame = cap.read()

            if frame is None:
                break

            if first_frame is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if COUNTER < NUM_OF_FRAMES_TO_STACK:
                    first_frames.append(gray)
                    COUNTER += 1
                    continue

                elif COUNTER == NUM_OF_FRAMES_TO_STACK:
                    first_frame = np.median(np.array(first_frames), axis=0).astype(np.uint8)
                    frames_dict.put({'first': first_frame})
                # frames_dict['first'] = firstFrame

            frames_dict.put({i: frame})
            i += 1
            if i % 50 == 0:
                print("read frame ", i)
        while not frames_dict.empty():
            pass

    # When everything done, release the capture
    cap.release()
