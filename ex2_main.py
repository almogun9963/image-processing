import numpy as np
import cv2
import ex2_utils
from matplotlib import pyplot as plt

print("Almog Amiga,211543285")
print("Ariel Yechezkel,211356449")






def test_conv1D():
    kernel = [1, -1, 0, 1, 0]
    arr = [0, 1, 0, 1, 0]
    ans = ex2_utils.conv1D(arr, kernel)
    ans2 = np.convolve(arr, kernel, 'full')
    print("my conv1D:", ans)
    print("np conv1D:", ans2)

def test_conv2D():

    img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cat.jpg", 1)
    kernel = np.array([[1 / 8, 1 / 8, 1 / 8], [1 / 8, 1 / 8, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    myconv2d = ex2_utils.conv2D(img, kernel)

    cvconv2d = cv2.filter2D(img, -1, kernel, cv2.BORDER_REPLICATE)

    plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(121), plt.imshow((myconv2d * 20).astype(np.uint8), cmap='gray')
    plt.title('my conv2D Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow((cvconv2d * 255).astype(np.uint8), cmap='gray')
    plt.title('opencv conv2D Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def test_convDerivative():
    img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cat.jpg", 1)
    direction, magnitude, der_x, der_y = ex2_utils.convDerivative(img)

    plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imshow((magnitude * 255).astype(np.uint8))
    plt.title('magnitude'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imshow(der_y )
    plt.title('der_y'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imshow(der_x )
    plt.title('der_x'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_edgeDetectionSobel():
    #img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cln1.png", 1)
    img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cat.jpg", 1)
    (cvsobel,mysobel ) = ex2_utils.edgeDetectionSobel(img, 1)

    plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(121), plt.imshow((mysobel* 255).astype(np.uint8), cmap='gray')
    plt.title('my sobel Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow((cvsobel* 255).astype(np.uint8), cmap='gray')
    plt.title('opencv sobel Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def test_edgeDetectionZeroCrossingSimple():
    img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cln1.png", 1)
    zcsimple = ex2_utils.edgeDetectionZeroCrossingSimple(img)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow((zcsimple * 255).astype(np.uint8), cmap='gray')
    plt.title('edge Detection Zero Crossing simple Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def test_edgeDetectionZeroCrossingLOG():
    #src = cv2.imread('C:\\Users\\Almog\\Downloads\\beach.jpg')
    #img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cln1.png",1)
    zclog = ex2_utils.edgeDetectionZeroCrossingLOG(img)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(zclog, cmap='gray')
    plt.title('edge Detection Zero Crossing using LOG Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def test_edgeDetectionCanny():
    img_path = 'C:\\Users\\Almog\\Downloads\\cln1.png'
    img = ex2_utils.imReadAndConvert(img_path, 1)
    mycanny, cvcanny = ex2_utils.edgeDetectionCanny(img, 0.3, 0.7)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(121), plt.imshow(mycanny, cmap='gray')
    plt.title('my canny Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cvcanny, cmap='gray')
    plt.title('opencv canny Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_houghCircle():
    circles = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\coins.jpg", 1)
    hc = ex2_utils.houghCircle(circles, 1, 10)
    cvhc = cv2.HoughCircles(circles, cv2.HOUGH_GRADIENT, minRadius=1, maxRadius=10)

    plt.imshow(circles, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(121), plt.imshow(hc, cmap='gray')
    plt.title('my hough Circle Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cvhc, cmap='gray')
    plt.title('opencv hough Circle Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def test():
    img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cln1.png", 1)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

    sobel = np.hypot(sobelx, sobely)
    plt.imshow(sobel, cmap='gray')
    plt.show()

if __name__ == '__main__':
    #test_conv1D()
    #test_conv2D()
    test_convDerivative()
    test_edgeDetectionSobel()
    #test_edgeDetectionZeroCrossingSimple()
    #test_edgeDetectionZeroCrossingLOG()
    test_edgeDetectionCanny()
    test_houghCircle()
    #test()
























