import ipywidgets as widgets
from IPython.display import display
from py.Session import Session
from py.ImageClassifier import AbstractImageClassifier
import numpy as np

# Old annotator script using IPython widgets, which are very slow and sometimes buggy.
# It is preferred to use quick_label.py.
class ImageAnnotator():
    def __init__(self, classifier: AbstractImageClassifier, session: Session, initial_scores = [], initial_annotations = [], load_from = None):
        self.scores = initial_scores
        self.annotations = initial_annotations
        self.score = -1
        self.classifier = classifier
        self.session = session

        if load_from is not None:
            data = np.load(load_from, allow_pickle=True)
            self.annotations = data[0]
            self.scores = data[1]

        normal_btn = widgets.Button(description = "Normal")
        anomalous_btn = widgets.Button(description = "Anomalous")
        self.button_box = widgets.HBox([normal_btn, anomalous_btn])
        self.output = widgets.Output(layout={"height": "400px"})
        display(self.button_box, self.output)
        normal_btn.on_click(self.mark_as_normal)
        anomalous_btn.on_click(self.mark_as_anomalous)
        self.next_image()
    
    # Click on normal button
    def mark_as_normal(self, _):
        with self.output:
            print("Marking as normal...")
        self.annotations.append(True)
        self.scores.append(self.score)
        self.next_image()
    
    # Click on anomalous button
    def mark_as_anomalous(self, _):
        with self.output:
            print("Marking as anomalous...")
        self.annotations.append(False)
        self.scores.append(self.score)
        self.next_image()
    
    # Show next image
    def next_image(self):
        img = self.session.get_random_motion_image(day_only=True)
        self.score = self.classifier.evaluate(img)
        self.output.clear_output()
        with self.output:
            display(img.to_ipython_image())
            print(f"score = {self.score}")

    # Save annotation data to file
    def save(self, filename: str):
        np.save(filename, [self.annotations, self.scores])