import cv2
import numpy as np
import time



NUM_OF_FRAMES_TO_STACK = 50
COUNTER = 0
first_frames = list()



def read(framesDict,path=0):
    cap = cv2.VideoCapture(path)
    firstFrame,i = None, 0
    print("Start reading video...")
    global COUNTER
    while (True):
        # Capture frame-by-frame
        while not framesDict.full():
            ret, frame = cap.read()

            if frame is None:
                break


            if firstFrame is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if COUNTER < NUM_OF_FRAMES_TO_STACK:
                    first_frames.append(gray)
                    COUNTER += 1
                    continue

                elif COUNTER == NUM_OF_FRAMES_TO_STACK:
                    firstFrame = np.median(np.array(first_frames), axis=0).astype(np.uint8)
                    framesDict.put({'first': firstFrame})
                # framesDict['first'] = firstFrame




            framesDict.put({i: frame})
            i+=1
            if i%50 ==0:
                print ("read frame ",i)
        while not framesDict.empty():
            pass


    # When everything done, release the capture
    cap.release()
