


import cv2
import numpy as np
import time
from multiprocessing import Process, Manager


from src.components.detector import detect
from src.components.drawer import draw
from src.components.streamer import read


def kill(p_array):
    for p in p_array:
        p.terminate()


def pipeline(path, transform_matrix=None):
    with Manager() as manager:

        # FRAMES = manager.dict()
        # DETECTIONS = manager.dict()

        FRAMES = manager.Queue(200)
        DETECTIONS = manager.Queue(200)

        processes = [Process(target=read, args=(FRAMES, path)),
                     Process(target=detect, args=(FRAMES, DETECTIONS)),
                     Process(target=draw, args=(DETECTIONS,transform_matrix))]

        for p in processes:
            p.start()
            time.sleep(1)

        for p in processes:
            p.join()
            p.kill()

        cv2.destroyAllWindows()
        # kill(processes)


def pipeline_seq(path):
    FRAMES = {}
    DETECTIONS = {}

    read(FRAMES, path)
    detect(FRAMES, DETECTIONS)
    draw(DETECTIONS)


if __name__ == '__main__':

    video_path = "/hdd/musashi/data/20190827/10.0.0.100_8000_33_7B83BDD312F945A69D0591D3A00981F4_/20190827_20190827145530_20190827145721_145531.mp4"
    matrix_path = "/hdd/musashi/matrix/20190827_20190827145530_20190827145721_145531.mp4.npy"
    trns_mat = np.load(matrix_path)
    pipeline(video_path,trns_mat)
