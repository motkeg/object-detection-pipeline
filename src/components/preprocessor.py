import cv2

DEFAULT_GAUSSIAN_KERNEL_SIZE = 21


def preprocess_frame(frame, gaussian_kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE):
    # convert to gray for detection:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gaussian_kernel_size > 0:
        frame = cv2.GaussianBlur(frame, (gaussian_kernel_size, gaussian_kernel_size), 0)

    return frame
