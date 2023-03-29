import taichi as ti
import cv2 as cv
from pathlib import Path

ti.init(arch=ti.cuda)


class ImageProvider:
    def __init__(self, basePath) -> None:
        self.path = basePath
        self.idx = 0
        pass
    def next(self):
        img_path = Path().joinpath(self.path, str(self.idx) + ".png")
        if Path(img_path).is_file():
            origin_mat = cv.imread(str(img_path), cv.IMREAD_UNCHANGED)
            gray_mat = cv.cvtColor(origin_mat, cv.COLOR_BGR2GRAY)
            self.idx += 1
            return gray_mat
        else:
            raise AssertionError("image path error")

    


if __name__ == "__main__":
    img_provider = ImageProvider("/home/garryling/workspace/taichiwork/taichi_lk/imgs")
    img1 = img_provider.next()
    while True:
        img2 = img_provider.next()
        img1_corners = cv.goodFeaturesToTrack(img1, 300, 0.3, 5)
        img2_corners, status, err = cv.calcOpticalFlowPyrLK(img1, img2, img1_corners, None, winSize=(16, 16), maxLevel=4)

        cnt = 0       
        for s in status:
            cnt += 1
        

