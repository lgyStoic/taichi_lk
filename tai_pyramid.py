import taichi as ti
import cv2 as  cv

img_rgb3_2d = ti.types.ndarray(ti.math.vec3, ndim=2)
img_gray_2d = ti.types.ndarray(ti.u8, ndim=2)
ti.init(arch=ti.cpu, debug = True, print_kernel_llvm_ir_optimized=True)
weight = ti.field(dtype=ti.f32, shape=5, offset=-2)

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

@ti.kernel
def img_with_pyrdown(img: img_gray_2d, img_blur: img_gray_2d, img_down: img_gray_2d):
    h, w = img.shape
    weight[-2] = 1
    weight[-1] = 4
    weight[0] = 6
    weight[1] = 4
    weight[2] = 1
    radius = 2
    total_weight = 16
    for i, j in ti.ndrange(h, w):
        l_begin, l_end = ti.max(0, i - radius), ti.min(h, i + radius + 1)
        total = 0.0
        for l in range(l_begin, l_end):
            wi= weight[l - i]
            total += img[l, j] * wi
        total /= total_weight
        img_blur[i,j] = ti.cast(total, ti.u8)
    
    for i, j in ti.ndrange(h, w):
        l_begin, l_end = ti.max(0, j - radius), ti.min(w, j + radius + 1)
        total = 0.0
        for l in range(l_begin, l_end):
            wi = weight[l - j]
            total += img_blur[i, l] * wi
        total /= total_weight
        img_blur[i,j] = ti.cast(total, ti.u8)

    for i, j in ti.ndrange(h // 2, w // 2):
        res = 0.0
        res += img[i * 2, j * 2]
        res += img[i * 2 + 1, j * 2] 
        res += img[i * 2, j * 2 + 1] 
        res += img[i * 2 + 1, j * 2 + 1]
        
        img_down[i, j] = ti.cast(res/4.0, ti.u8)


@ti.kernel
def tai_pyramid(img: img_gray_2d, level: int, win_size: int):
    # pad image with window size 
    pass


if __name__ == "__main__":
    # ti.init(arch=ti.cuda)
    img:img_gray_2d = cv.imread("./imgs/0.png", cv.IMREAD_GRAYSCALE)
    pad_size = 50
    h, w = img.shape
    img_res:img_gray_2d = ti.ndarray(ti.uint8, shape=(h + 2 * pad_size, w + 2 * pad_size))
    img_with_pad(img, img_res, pad_size)
    cv.imwrite("res.pgm", img_res.to_numpy())

    img_pad_cv = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_REPLICATE)
    cv.imwrite("res_cv.pgm", img_pad_cv)

    img_blur:img_gray_2d = ti.ndarray(ti.uint8, shape=(h + 2 * pad_size, w + 2 * pad_size))
    img_res_div_2:img_gray_2d = ti.ndarray(ti.uint8, shape=(h // 2 + pad_size, w // 2 + pad_size))
    img_with_pyrdown(img_res, img_blur,img_res_div_2)
    cv.imwrite("res_div2.pgm", img_res_div_2.to_numpy())
    print("done")
    # pyrs = prepare(h, w, 3)
    # tai_pyramid()