import cv2
import numpy as np

from src.components.preprocessor import preprocess_frame

NUM_OF_FRAMES_FOR_MOG = 50
NUM_OF_FRAMES_TO_STACK = 50
ERODE_DILATE_ITERATION = 3
COUNTER = 0
first_frames = list()


def get_contours_from_MOG(bg_model, frame, learning_rate,filter_shadow=False):
    fgmask = bg_model.apply(frame, learningRate=learning_rate)
    threshold_frame = cv2.erode(fgmask,None,iterations=ERODE_DILATE_ITERATION)
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=ERODE_DILATE_ITERATION)
    if filter_shadow:
        threshold_frame = ((threshold_frame == 255)*255).astype('uint8')
    cnts = cv2.findContours(threshold_frame, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    return threshold_frame, cnts


def get_contours_from_simple_subtraction(first_frame, frame):
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(first_frame, frame)
    threshold_frame = cv2.threshold(frameDelta, 255, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
    cnts = cv2.findContours(threshold_frame, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    return threshold_frame, cnts


def detect(frameDict, detectionsDict):
    print("Strat detecting...")
    firstFrame = None
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    learning_rate = 1

    global COUNTER
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

            processed_frame = preprocess_frame(frame)

            simple_tresh_frame, simple_cnts = get_contours_from_simple_subtraction(firstFrame, processed_frame)
            mog_tresh_frame, mog_cnts = get_contours_from_MOG(fgbg, processed_frame, learning_rate,filter_shadow=True)

            tresh_stack = np.hstack((simple_tresh_frame, mog_tresh_frame))
            # show the threshold frame from the simple subtraction and MOG
            cv2.imshow("Simple Delta | MOG Delta", cv2.resize(tresh_stack, (1920, 768)))
            # cv2.imshow("First Frame", cv2.resize(firstFrame, (960, 768)))
            if cv2.waitKey(10) & 0xFF == 27:
                break

            if not detectionsDict.full():
                detectionsDict.put({key: (frame, mog_cnts[0])})
