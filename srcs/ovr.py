from dataset import Dataset
from gd import GradientDescent, Batch, MiniBatch, Stochastic
from regression import LogisticRegression


class OneVsRest:
    
    MODEL = LogisticRegression
    ALGORITHM = MiniBatch
    LEARNING_RATE = 0.1
    EPOCHS = 100
    DEBUG = True
    
    def __init__(self, data, classes) -> None:
        
        self.data = data
        self.classes = classes
        self.models = {}

    def build_models(self):

        for c in self.classes:
            
            # for each class, we build a logreg model -> C vs not C
            
            pass