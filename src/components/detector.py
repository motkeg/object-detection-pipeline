import cv2
import numpy as np

NUM_OF_FRAMES_FOR_MOG = 50
NUM_OF_FRAMES_TO_STACK = 50
COUNTER = 0
first_frames = list()


def get_contours_from_MOG(bg_model, frame, learning_rate):
    fgmask = bg_model.apply(frame, learningRate=learning_rate)
    threshold_frame = cv2.dilate(fgmask, None, iterations=2)
    cnts = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    return threshold_frame, cnts


def get_contours_from_simple_subtraction(first_frame, frame):
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(first_frame, frame)
    threshold_frame = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
    cnts = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    return (threshold_frame, cnts)


def detect(frameDict, detectionsDict):
    print("Strat detecting...")
    firstFrame = None
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    learning_rate = 1

    global COUNTER
    while True:

        while True:

            val = frameDict.get()
            for key, frame in val.items():

                if key == 'first':  # skip the first gray frame
                    firstFrame = frame
                    break
                if firstFrame is None:
                    break

                if int(key) % 50 == 0:
                    print("detect frame ", key)

                if int(key) > 50:
                    learning_rate = 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                simple_tresh_frame, simple_cnts = get_contours_from_simple_subtraction(firstFrame, gray)
                mog_tresh_frame, mog_cnts = get_contours_from_MOG(fgbg, gray, learning_rate)

                tresh_stack = np.hstack((simple_tresh_frame, mog_tresh_frame))

                # cv2.imshow("Simple Delta frame",simple_tresh_frame)
                cv2.imshow("Simple Delta | MOG Delta", cv2.resize(tresh_stack, (1920, 768)))
                cv2.imshow("First Frame", cv2.resize(firstFrame, (960, 768)))
                if cv2.waitKey(10) & 0xFF == 27:
                    break

                # cnts = imutils.grab_contours(mog_cnts)
                if not detectionsDict.full():
                    detectionsDict.put({key: (frame, mog_cnts[0])})
