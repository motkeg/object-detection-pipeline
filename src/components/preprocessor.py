import cv2

DEFAULT_IS_CONVERT_TO_GRAY = True
DEFAULT_GAUSSIAN_KERNEL_SIZE = 21


def preprocess_frame(frame, is_convert_to_gray=DEFAULT_IS_CONVERT_TO_GRAY,
                     gaussian_kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE):
    if is_convert_to_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gaussian_kernel_size > 0:
        frame = cv2.GaussianBlur(frame, (gaussian_kernel_size, gaussian_kernel_size), 0)
    return frame
