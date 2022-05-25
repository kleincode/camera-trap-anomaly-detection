from py.Session import MotionImage

class AbstractImageClassifier():
    def evaluate(self, motion_img: MotionImage, display=False) -> int:
        raise NotImplementedError("Please implement evaluate(motion_img)!")