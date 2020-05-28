import numpy as np
import cv2
import math
from skimage import filters
import scipy.ndimage as nd
import scipy
from scipy.ndimage import gaussian_filter


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:

    if representation == 1:
        img = cv2.imread(filename, 0)
        arr = np.asarray(img)

    elif representation == 2:
        img = cv2.imread(filename)
        RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.asarray(RGB)

    else:
        return None

    normelize = arr.min()+((arr-arr.min())/(arr.max()-arr.min()))
    return normelize

def conv_trans1d(image):
    image_copy = image.copy()

    for i in range(image.shape[0]):
        image_copy[i] = image[image.shape[0]-i-1]

    return image_copy


"""
Convolve a 1-D array with a given kernel
:param inSignal: 1-D array
:param kernel1: 1-D array as a kernel
:return: The convolved array
"""
#https://www.youtube.com/watch?v=BPBTmXKtFRQ
def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
    arr = []
    sum = 0
    copied = np.resize(inSignal, len(inSignal) + len(kernel1) - 1)
    i = len(kernel1) - 1
    while (len(copied) > i):
        copied[i] = 0
        i += 1

    for j in range(len(inSignal) + len(kernel1) - 1):
        for k in range(len(inSignal) - 1):
            sum = sum + kernel1[k] * copied[j - k]
        arr.append(sum)
        sum = 0

    return arr


def conv_trans2d(image):
    image_copy = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]

    return image_copy

"""
Convolve a 2-D array with a given kernel
:param inImage: 2D image
:param kernel2: A kernel
:return: The convolved image
"""
#https://www.youtube.com/watch?v=BPBTmXKtFRQ
def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    image_h = inImage.shape[0]
    image_w = inImage.shape[1]
    image_conv = np.zeros(inImage.shape)
    if np.size(np.shape(kernel2)) == 1:
        kernel = conv_trans1d(kernel2)
        kernel_h = kernel.shape[0]
        kernel_w = 1

        h = kernel_h//2
        w = kernel_w//2



        for i in range(h,image_h - h):
            for j in range(w, image_w - w):
                sum = 0

                for m in range(kernel_h):
                    for n in range(1):
                        sum = sum + kernel[m] + inImage[i - h + m][j - w + n]

                image_conv[i][j] = sum

    if np.size(np.shape(kernel2)) == 2:
        kernel = conv_trans2d(kernel2)
        kernel_h = kernel.shape[0]
        kernel_w = kernel.shape[1]

        h = kernel_h // 2
        w = kernel_w // 2

        # image_conv = np.zeros(inImage.shape)

        for i in range(h, image_h - h):
            for j in range(w, image_w - w):
                sum = 0

                for m in range(kernel_h):
                    for n in range(kernel_w):
                        sum = sum + kernel[m][n] + inImage[i - h + m][j - w + n]

                image_conv[i][j] = sum


    return image_conv



"""
Calculate gradient of an image
:param inImage: Grayscale iamge
:return: (directions, magnitude,x_der,y_der)
"""
# https://stackoverflow.com/questions/49732726/how-to-compute-the-gradients-of-image-using-python
def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    new_img = np.copy(inImage)

    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.transpose(kernelx)

    #kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    #kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    x_der = cv2.filter2D(new_img, -1, kernelx)
    y_der = cv2.filter2D(new_img, -1, kernely)

    magnitude = np.zeros(np.shape(new_img))
    directions = np.zeros(np.shape(new_img))
    h_img, w_img = np.shape(new_img)
    for i in range(h_img - 1):
        for j in range(w_img-1):
            magnitude[i][j] = math.sqrt(pow(x_der[i][j], 2) + pow(y_der[i][j], 2))
            directions[i][j] = math.degrees(math.atan((np.divide(y_der[i][j], x_der[i][j], where=x_der[i][j] != 0))))
    return directions, magnitude, x_der, y_der




def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g

"""
Blur an image using a Gaussian kernel
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""
#https://medium.com/spinor/a-straightforward-introduction-to-image-blurring-smoothing-using-python-f8870cf1096
def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    size = len(kernel_size)
    gaussian_filter_img = conv2D(in_image, gaussian_kernel(size, sigma=3))


    return gaussian_filter_img





"""
Blur an image using a Gaussian kernel using OpenCV built-in functions
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""
#https://medium.com/spinor/a-straightforward-introduction-to-image-blurring-smoothing-using-python-f8870cf1096
def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    size = len(kernel_size)
    #blur = cv2.blur(in_image, (rows, cols))
    blur = cv2.GaussianBlur(in_image, (5, 5), sigmaX=size)
    return blur




def norm(img1: np.ndarray, img2: np.ndarray , thresh: float):
    img_copy = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            q = (img1[i][j]**2 + img2[i][j]**2)**(1/2)
            if(q > thresh):
                img_copy[i][j] = 255
            else:
                img_copy[i][j] = q
    return img_copy


"""
Detects edges using the Sobel method
:param img: Input image
:param thresh: The minimum threshold for the edge response
:return: opencv solution, my implementation
"""
#https://www.youtube.com/watch?v=Ie2Tj_3Ug2A
def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)-> (np.ndarray, np.ndarray):

    kernel = np.zeros(shape = (3,3))
    kernel[0,0] = -1
    kernel[0, 1] = -2
    kernel[0, 2] = -1
    kernel[1, 0] = 0
    kernel[1, 1] = 0
    kernel[1, 2] = 0
    kernel[2, 0] = 1
    kernel[2, 1] = 2
    kernel[2, 2] = 1
    if(thresh == 1):
        kernel = kernel *1/3
    gy = cv2.filter2D(img, -1, kernel)

    kernel2 = np.zeros(shape=(3, 3))
    kernel2[0, 0] = -1
    kernel2[0, 1] = 0
    kernel2[0, 2] = 1
    kernel2[1, 0] = -1
    kernel2[1, 1] = 0
    kernel2[1, 2] = 1
    kernel2[2, 0] = -2
    kernel2[2, 1] = 0
    kernel2[2, 2] = 2
    if (thresh == 1):
        kernel2 = kernel2 *1/3
    gx = cv2.filter2D(img, -1, kernel2)

    myans = norm(gx,gy,thresh)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=thresh)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=thresh)

    sobel = np.hypot(sobelx, sobely)

    return (sobel,myans)


"""
Detecting edges using the "ZeroCrossing" method
:param I: Input image
:return: Edge matrix
"""
#https://books.google.co.il/books?id=-9_dDwAAQBAJ&pg=PA73&lpg=PA73&dq=edge+detection+zero+crossing+python+code+implement&source=bl&ots=TXm8NUN-GG&sig=ACfU3U0x7QALvrgSGZ1oGEnFPpqx6pjufw&hl=iw&sa=X&ved=2ahUKEwjHhoWc1M_pAhUGcBQKHRLmA6MQ6AEwEXoECAoQAQ#v=onepage&q=edge%20detection%20zero%20crossing%20python%20code%20implement&f=false
#https://www.mathworks.com/help/images/ref/edge.html
def edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    #we were not sure if the color that should be inside is the black one or the white one
    #we took the white inside
    sample = gaussian_filter(img, sigma=2)
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 8, -1],
    #                    [-1, -1, -1]])
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    # kernel3 = np.multiply(kernel , kernel2)
    answer = cv2.filter2D(sample, -1, kernel)
    return answer



"""
Detecting edges using the "ZeroCrossingLOG" method
:param I: Input image
:return: :return: Edge matrix
"""
#https://github.com/debikadutt/Edge-Detection-using-LoG-and-DoG/blob/master/LoG.py#L8
def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    LoG = nd.gaussian_laplace(img, 2)
    thres = np.absolute(LoG).mean() * 0.75
    output = scipy.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y - 1:y + 2, x - 1:x + 2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1

    return  output


"""
Detecting edges usint "Canny Edge" method
:param img: Input image
:param thrs_1: T1
:param thrs_2: T2
:return: opencv solution, my implementation
"""
# http://www.adeveloperdiary.com/data-science/computer-vision/implement-canny-edge-detector-using-python-from-scratch/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
# https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html
def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    filltered = gaussian_filter(img, sigma=3)

    first = (filltered * 255).astype(np.uint8)
    cvans = cv2.Canny(first, thrs_1, thrs_2)

    second = filltered * 50

    directions, magnitude, x_der, y_der = convDerivative(second)
    magnitude = non_max_suppression(magnitude, x_der)

    myans = hysteresis(magnitude, thrs_1, thrs_2)

    return cvans, myans

# https://github.com/metinmertakcay/Canny_Edge_Detection/blob/master/main.py
def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction<= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    return output

def hysteresis(image, weak=50, strong=255):
    print("- Hysteresis operation -")
    output_image = copy(image)
    for i in range(1, len(image) - 1):
        for j in range(1, len(image[0]) - 1):
            if (image[i][j] == weak):
                if ((image[i+1][j-1] == strong) or (image[i+1][j] == strong) or (image[i+1][j+1] == strong)
                 or (image[i][j-1] == strong) or (image[i][j+1] == strong)
                 or (image[i-1][j-1] == strong) or (image[i-1][j] == strong) or (image[i-1][j+1] == strong)):
                    output_image[i][j] = strong
                else:
                    output_image[i][j] = 0
    return output_image

def copy(image):
    copy_image = []
    for i in range(len(image)):
        copy_image_col = []
        for j in range(len(image[0])):
            copy_image_col.append(image[i][j])
        copy_image.append(copy_image_col)
    return copy_image


#https://en.wikipedia.org/wiki/Circle_Hough_Transform
#https://github.com/PavanGJ/Circle-Hough-Transform/blob/master/main.py
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
# def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
#     #circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, minRadius=min_radius, maxRadius=max_radius)
#     #canny = cv2.Canny(img)
#
#     #voting = np.zeros(shape=(canny.shape[0],canny.shape[1] ,max_radius - min_radius ))
#     #for x in range(canny.shape[0]):
#     #    for y in range(canny.shape[1]):
#     #        for radius in (min_radius,max_radius):
#     #            for theta in(0,360):
#
#     #                a = x - radius * radius * math.cos(theta * math.pi / 180);
#     #                b = y - radius * math.sin(theta * math.pi / 180);
#     #                voting[a,b,radius] +=1
#
#     region = 15
#     threshold = 8.1
#
#     res = smoothen(img)
#     res = edge(res, 128)
#     (M, N) = res.shape
#     R = max_radius - min_radius
#     A = np.zeros((max_radius, M + 2 * max_radius, N + 2 * max_radius))
#     B = np.zeros((max_radius, M + 2 * max_radius, N + 2 * max_radius))
#     theta = np.arange(0, 360) * np.pi / 180
#     edges = np.argwhere(res[:, :])
#     for val in range(R):
#         r = min_radius + val
#
#         bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
#         (m, n) = (r + 1, r + 1)
#         for angle in theta:
#             x = int(np.round(r * np.cos(angle)))
#             y = int(np.round(r * np.sin(angle)))
#             bprint[m + x, n + y] = 1
#         constant = np.argwhere(bprint).shape[0]
#         for x, y in edges:
#             X = [x - m + max_radius, x + m + max_radius]
#             Y = [y - n + max_radius, y + n + max_radius]
#             A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
#         A[r][A[r] < threshold * constant / r] = 0
#
#     for r, x, y in np.argwhere(A):
#         temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
#         try:
#             p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
#         except:
#             continue
#         B[r + (p - region), x + (a - region), y + (b - region)] = 1
#
#     return B[:, max_radius:-max_radius, max_radius:-max_radius]
#
#
#
# def smoothen(img):
#
#     gaussian = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])
#     img = np.array(img)
#     smooth = cv2.filter2D(img,-1,gaussian)
#
#     return smooth
#
# def edge(img,threshold):
#
#     laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
#     sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#
#     G_x = cv2.filter2D(img,-1,sobel)
#
#     G_y = cv2.filter2D(img,-1,sobel.transpose())
#
#     G = pow((G_x*G_x + G_y*G_y),0.5)
#
#     G[G<threshold] = 0
#     L = cv2.filter2D(img,-1,laplacian)
#     if L is None:
#         return
#     (M,N) = L.shape
#
#     temp = np.zeros((M+2,N+2))
#     temp[1:-1,1:-1] = L
#     result = np.zeros((M,N))
#     for i in range(1,M+1):
#         for j in range(1,N+1):
#             if temp[i,j]<0:
#                 for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
#                         if temp[i+x,j+y]>0:
#                             result[i-1,j-1] = 1
#
#     return img

"""
Find Circles in an image using a Hough Transform algorithm extension
:param I: Input image
:param minRadius: Minimum circle radius
:param maxRadius: Maximum circle radius
:return: A list containing the detected circles,
[(x,y,radius),(x,y,radius),...]
"""

def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:

  