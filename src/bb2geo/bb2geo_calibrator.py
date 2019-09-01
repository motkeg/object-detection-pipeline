import cv2
import os
import numpy as np
import argparse

from src.bb2geo.utils import BoxDrawerUtil
from tests.bb2geo_tests import load_matrix_test

DEFAULT_THICKNESS = 2


def build_transform_matrix(stream_path, map_path, output_path="./"):
    stream_points = None
    map_points = None
    name = stream_path.split('/')[-1]
    full_output_path = os.path.join(output_path, "{}.npy".format(name))

    for path in [stream_path, map_path]:

        cap = cv2.VideoCapture(path)
        while cap:
            success, img = cap.read()

            if stream_points is None:
                stream_points = BoxDrawerUtil.select_bbox_area(img).copy()
            else:
                map_points = BoxDrawerUtil.select_bbox_area(img).copy()

            key = cv2.waitKey(20)
            if key & 0xFF == 27:
                cv2.destroyAllWindows()
                break
            break
    print(stream_points)
    print(map_points)

    stream_points = np.array(stream_points, dtype="float32")
    map_points = np.array(map_points, dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(stream_points, map_points)
    np.save(full_output_path, transform_matrix)
    print("Matrix saved in: {}".format(full_output_path))

    return full_output_path


def main():
    parser = argparse.ArgumentParser(description='bb2geo calibrator')
    parser.add_argument("--stream_path", "-s", help="string - the rtsp stream path", required=True)
    parser.add_argument("--map_path", "-m", help="string - path to map file", required=True)
    parser.add_argument("--output_path", "-o", help="string - path of output dir", required=False)

    args = parser.parse_args()

    full_output_path = build_transform_matrix(args.stream_path,args.map_path,args.output_path)

    load_matrix_test(full_output_path, args.map_path)


if __name__ == '__main__':
    main()
