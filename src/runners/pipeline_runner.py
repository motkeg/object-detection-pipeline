import time
from multiprocessing import Process, Manager

import cv2

from src.components.detector import detect
from src.components.drawer import draw
from src.components.streamer import read


def kill(p_array):
    for p in p_array:
        p.terminate()


def pipeline(path):
    with Manager() as manager:

        # FRAMES = manager.dict()
        # DETECTIONS = manager.dict()

        FRAMES = manager.Queue(200)
        DETECTIONS = manager.Queue(200)

        processes = [Process(target=read, args=(FRAMES, path)),
                     Process(target=detect, args=(FRAMES, DETECTIONS)),
                     Process(target=draw, args=(DETECTIONS,))]

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
    # path = '/hdd/mosashi-vmd/videos/20190701_20190701155751_20190701155919_155751.mp4'
    # path = "/hdd/mosashi-vmd/videos/20190701_20190701155755_20190701155922_155754.mp4"
    path = "/hdd/musashi/data/20190827/10.0.0.100_8000_33_7B83BDD312F945A69D0591D3A00981F4_/20190827_20190827150002_20190827150136_150003.mp4"
    # pipline_seq(path)
    pipeline(path)
