import imgpreproc.resizing as resizing
import imgpreproc.reading as reading
import numpy as np
from PIL import Image


def main():
    """
    images = reading.read_from_server(url_dir="Cellprofiler%20workshop/Bob/FN1/")
    for i in images:
        Image.fromarray(i).show()
    """
    # images = reading.read_local("/home/sari/Desktop/sk_cp_output")
    images = reading.read_local("/home/sari/Desktop/sk_cp_output")
    img = images[0]
    Image.fromarray(resizing.resize(img, 150, 150)).show()
    resized_img = resizing.resize2(img, 10, 10)
    print type(resized_img)
    # arr = np.array([[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26, 27], [31, 32, 33, 34, 35, 36, 37]])
    # resized_arr = resizing.sliding_window(arr, 3, 3)

    print "This is the number of results" + str(len(resized_img))
    if type(resized_img) == np.ndarray:
        Image.fromarray(resizing.resize(resized_img, 150, 150)).show()
    else:
        for i in resized_img:
            Image.fromarray(resizing.resize(i, 100, 100)).show()


    """
    resized_img = resizing.resize(img, 100, 100, Image.NEAREST)
    resized_img1 = resizing.resize(img, 100, 100, Image.BILINEAR)
    resized_img2 = resizing.resize(img, 100, 100, Image.BICUBIC)
    resized_img3 = resizing.resize(img, 100, 100, Image.LANCZOS)
    Image.fromarray(img).save("/home/sari/Desktop/orig.tiff")
    Image.fromarray(resized_img).show()
    Image.fromarray(resized_img).save("/home/sari/Desktop/nn.tiff")
    Image.fromarray(resized_img1).show()
    Image.fromarray(resized_img1).save("/home/sari/Desktop/bl.tiff")
    Image.fromarray(resized_img2).show()
    Image.fromarray(resized_img2).save("/home/sari/Desktop/bic.tiff")
    Image.fromarray(resized_img3).show()
    Image.fromarray(resized_img3).save("/home/sari/Desktop/lanc.tiff")
    """

if __name__ == '__main__':
    main()
