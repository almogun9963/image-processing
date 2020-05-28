import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

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

    img = ex2_utils.imReadAndConvert("Giraffe.jpg", 1)
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
    img = ex2_utils.imReadAndConvert("lena.jpg", 1)
    direction, magnitude, der_x, der_y = ex2_utils.convDerivative(img)

    plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imshow((magnitude * 255).astype(np.uint8), cmap='gray')
    plt.title('magnitude'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imshow(der_y , cmap='gray')
    plt.title('der_y'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imshow(der_x , cmap='gray')
    plt.title('der_x'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_edgeDetectionSobel():
    #img = ex2_utils.imReadAndConvert("C:\\Users\\Almog\\Downloads\\cln1.png", 1)
    img = ex2_utils.imReadAndConvert("lena.jpg", 1)
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
    img = ex2_utils.imReadAndConvert("Giraffe.jpg", 1)
    zcsimple = ex2_utils.edgeDetectionZeroCrossingSimple(img)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow((zcsimple * 255).astype(np.uint8), cmap='gray')
    plt.title('edge Detection Zero Crossing simple Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def test_edgeDetectionZeroCrossingLOG():
    img = ex2_utils.imReadAndConvert("monaliza.jpg",1)
    zclog = ex2_utils.edgeDetectionZeroCrossingLOG(img)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(zclog, cmap='gray')
    plt.title('edge Detection Zero Crossing using LOG Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def test_edgeDetectionCanny():
    img = ex2_utils.imReadAndConvert("Giraffe.jpg", 1)
    cvcanny, mycanny = ex2_utils.edgeDetectionCanny(img, 0.3, 0.7)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(121), plt.imshow(mycanny, cmap='gray')
    plt.title('my canny Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cvcanny, cmap='gray')
    plt.title('opencv canny Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_houghCircle():

    img = ex2_utils.imReadAndConvert("coins.jpg", 1)
    smoothed_img = gaussian_filter(img, sigma=2)
    smoothed_img = (smoothed_img * 255).astype(np.uint8)
    edged_image = cv2.Canny(smoothed_img, 100, 200)

    circles = ex2_utils.houghCircle(edged_image, 1,10)

    for ver in circles:
        plt.imshow((ver * 255).astype(np.uint8), cmap='gray')

    plt.show()



def test_blurImage1():
    img = ex2_utils.imReadAndConvert("opencv.png", 1)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    blurred = ex2_utils.blurImage1(img,kernel)
    plt.subplot(121),plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blurred, cmap='gray')
    plt.title('blurred Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def test_blurImage2():
    img = ex2_utils.imReadAndConvert("opencv.png", 1)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])*1/16
    blurred = ex2_utils.blurImage2(img,kernel)
    plt.subplot(121),plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blurred, cmap='gray')
    plt.title('blurred Image'), plt.xticks([]), plt.yticks([])
    plt.show()




def main():
    test_conv1D()
    test_conv2D()
    test_convDerivative()
    test_blurImage1()
    test_blurImage2()
    test_edgeDetectionSobel()
    test_edgeDetectionZeroCrossingSimple()
    test_edgeDetectionZeroCrossingLOG()
    test_edgeDetectionCanny()
    test_houghCircle()

if __name__ == '__main__':
    main()



