from py.Session import MotionImage

# Abstract class which represents an image classifier.
# Returns a real number for any image which can then be thresholded.
class AbstractImageClassifier():
    def evaluate(self, motion_img: MotionImage, display=False) -> int:
        raise NotImplementedError("Please implement evaluate(motion_img)!")