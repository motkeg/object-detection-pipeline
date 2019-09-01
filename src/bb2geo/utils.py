import cv2
from itertools import chain
from random import randint



def create_colors_list(num_of_colors):
    """
    creates a list of different "num_of_colors" colors

    :param num_of_colors: The number of different colors to generate
    :return: a list of "num_of_colors" tuples. Each tuple represents a different color in bgr or rgb mode.
    """
    diff_colors = []
    for i in range(num_of_colors):
        b = randint(0, 255)
        g = randint(0, 255)
        r = randint(0, 255)

        diff_colors.append((b, g, r))

    return diff_colors



class BoxDrawerUtil(object):

    DEFAULT_THICKNESS = 2
    points = []
    img = None

    @staticmethod
    def click(event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting (x, y)
        # coordinates and indicate that cropping is being performed

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(BoxDrawerUtil.img, (x, y), 5, (255, 0, 0), thickness=3)
            BoxDrawerUtil.points.extend([[x, y]])

    @staticmethod
    def select_bbox_area(image,tuple=True):
        BoxDrawerUtil.img = image.copy()
        BoxDrawerUtil.points.clear()
        cv2.namedWindow('select points')
        cv2.setMouseCallback('select points', BoxDrawerUtil.click)
        cv2.imshow('select points', BoxDrawerUtil.img)
        while 1:

            cv2.imshow('select points', BoxDrawerUtil.img)
            # cv2.imwrite("/hdd/musashi/data/warehouse setup_27_08_19_1920X1080.jpg",cv2.resize(BoxDrawerUtil.img,(1920,1080)))
            if cv2.waitKey(20) & 0xFF == 27: # Esc button
                cv2.destroyAllWindows()
                break

        if tuple:
            return BoxDrawerUtil.points
        else:
            return list(chain.from_iterable(BoxDrawerUtil.points))


