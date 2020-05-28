authors: Almog Amiga 211543285
	 Ariel Yechezkel 211356449

we wrote this project in pycharm

the project contain 2 files :
1 - ex2_utils.py - where we implement all the fucntions
2 - ex2_main.py - where we run and test all the fucntions we wrote in ex2_utils.py

here is all the function we implement plus a short explenation:

def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
"""
Convolve a 1-D array with a given kernel
:param inSignal: 1-D array
:param kernel1: 1-D array as a kernel
:return: The convolved array
"""


def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
"""
Convolve a 2-D array with a given kernel
:param inImage: 2D image
:param kernel2: A kernel
:return: The convolved image
"""


def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
"""
Calculate gradient of an image
:param inImage: Grayscale iamge
:return: (directions, magnitude,x_der,y_der)
"""


def blurImage1(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
"""
Blur an image using a Gaussian kernel
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""


def blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
"""
Blur an image using a Gaussian kernel using OpenCV built-in functions
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)
-> (np.ndarray, np.ndarray):
"""
Detects edges using the Sobel method
:param img: Input image
:param thresh: The minimum threshold for the edge response
:return: opencv solution, my implementation
"""


def edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray)
"""
Detecting edges using the "ZeroCrossing" method
:param I: Input image
:return: Edge matrix
"""


def edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray)
"""
Detecting edges using the "ZeroCrossingLOG" method
:param I: Input image
:return: :return: Edge matrix
"""


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float)
-> (np.ndarray, np.ndarray):
"""
Detecting edges usint "Canny Edge" method
:param img: Input image
:param thrs_1: T1
:param thrs_2: T2
:return: opencv solution, my implementation
"""


def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list
"""
Find Circles in an image using a Hough Transform algorithm extension
:param I: Input image
:param minRadius: Minimum circle radius
:param maxRadius: Maximum circle radius
:return: A list containing the detected circles,
[(x,y,radius),(x,y,radius),...]
"""




