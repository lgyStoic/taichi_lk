import taichi as ti
import cv2 as  cv

img_rgb3_2d = ti.types.ndarray(ti.math.vec3, ndim=2)
img_gray_2d = ti.types.ndarray(ti.u8, ndim=2)

def prepare(h, w, level):
    pyrs = []
    threshold  = min(h, w)
    for i in range(level):
        img_l = ti.ndarray(dtype=ti.u8, shape=(h // 2, w // 2))
        img_grad = ti.ndarray(dtype=ti.types.vector(2, ti.int16), shape=(h // 2, w // 2))
        pyrs.append([img_l, img_grad])
        h = h // 2
        w = w // 2
        if h <= threshold or w <= threshold:
            break
    return pyrs

@ti.kernel
def img_with_pad(img: img_gray_2d, img_pad: img_gray_2d, pad_size:int):
    h, w = img.shape
    #for i, j in ti.ndrange(h + 2 * pad_size, w + 2 * pad_size):
    #    if i >= pad_size and i < h + pad_size and j >= pad_size and j < w + pad_size:
    #        img_pad[i, j] = img[i - pad_size, j - pad_size] 
    #    elif i < pad_size:
    #        img_pad[i, j] = img[0, j - pad_size]
    #    elif j < pad_size:
    #        img_pad[i, j] = img[i - pad_size, 0]
    #    elif i >= pad_size + h:
    #        img_pad[i, j] = img[h - 1, j - pad_size]
    #        # img_pad[i, j] = 255
    #    elif j >= pad_size + w:
    #        img_pad[i, j] = img[i - pad_size, w - 1]
    #        # img_pad[i, j] = 255

    for i, j in ti.ndrange(h, w):
        img_pad[i + pad_size, j + pad_size] = img[i, j]
    
    #left
    for i in ti.ndrange((0, pad_size)):
        for j in ti.ndrange((pad_size, h + pad_size)):
            img_pad[j, i] = img_pad[j, pad_size]
    #right
    for i in ti.ndrange((pad_size + w - 1, w + 2 * pad_size)):
        for j in ti.ndrange((pad_size, h + pad_size)):
            img_pad[j, i] = img_pad[j, w - 1 + pad_size]
    #top
    for i in ti.ndrange((0, pad_size)):
        for j in ti.ndrange((0, pad_size * 2 + w)):
            img_pad[i, j] = img_pad[pad_size, j]
    #down
    for i in ti.ndrange((h + pad_size, h + 2* pad_size)):
        for j in ti.ndrange((0, pad_size * 2 + w)):
            img_pad[i, j] = img_pad[h - 1 + pad_size, j]
        

# same pyramid as opencv
#@ti.kernel
#def tai_pyramid(img: img2d, level: int, win_size: int):
#    # pad image with window size 
#    
#    pass


if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    img:img_gray_2d = cv.imread("./imgs/0.png", cv.IMREAD_GRAYSCALE)
    pad_size = 50
    h, w = img.shape
    img_res:img_gray_2d = ti.ndarray(ti.uint8, shape=(h + 2 * pad_size, w + 2 * pad_size))
    img_with_pad(img, img_res, pad_size)
    cv.imwrite("res.pgm", img_res.to_numpy())

    img_pad_cv = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_REPLICATE)
    cv.imwrite("res_cv.pgm", img_pad_cv)
    # pyrs = prepare(h, w, 3)
    # tai_pyramid()