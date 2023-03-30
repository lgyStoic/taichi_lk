import taichi as ti

img2d = ti.types.ndarray(ti.math.vec3, ndim=2)

# same pyramid as opencv
@ti.kernel
def tai_pyramid(img: img2d, level: int, win_size: int):
    # pad image with window size 
    
    pass


if __name__ == "__main__":
    tai_pyramid()