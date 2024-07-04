from matplotlib import pyplot
import numpy as np
import random

from dataset import Dataset
from gd import GradientDescent, Batch, MiniBatch, Stochastic
from regression import LogisticRegression
from stats import Statistics

class OneVsRestClassifier:
    
    MODEL = LogisticRegression
    ALGORITHM = Stochastic
    LEARNING_RATE = 0.5
    EPOCHS = 50
    DEBUG = True
    SEED = 42
    
    def __init__(self, dataset: Dataset, target: str, feature_per_class: dict[str, str]) -> None:
        
        random.seed(self.SEED)
        
        self.dataset = dataset
        self.feature_per_class = feature_per_class
        self.models = {c : GradientDescent(model=self.MODEL,
                                           algorithm=self.ALGORITHM,
                                           learning_rate=self.LEARNING_RATE,
                                           epochs=self.EPOCHS,
                                           init_weight=1,
                                           init_bias=1,
                                           debug=self.DEBUG)
                       for c in self.feature_per_class.keys()}
        self.target = target

    def fit_models(self):
        
        def normalize_x(data: list[tuple[float, float]]) -> list[tuple[float, float]]:

            x = [row[0] for row in data]

            x_mean =  Statistics.mean(x)
            x_std = Statistics.std(x)
            
            data = [((row[0] - x_mean) / x_std, row[1]) for row in data]
            
            return data

        for c in self.feature_per_class.keys():
            
            print(f"Fitting class {c}: {self.feature_per_class[c]}")
            
            # for each class, we build a logreg model -> C vs not C
            
            self.dataset.clean(target_feature=self.target,
                               include_features=[self.feature_per_class[c]])
            
            encode = lambda target: 1 if target == c else 0
            
            data = self.dataset.get_cleaned_data()
            target = self.dataset.get_target()
            data = [(row[0], encode(target[i])) for i, row in enumerate(data)]
            data = normalize_x(data)
            
            self.models[c].fit(data)
            print(f"weight: {self.models[c].get_weight()}")
            print(f"bias: {self.models[c].get_bias()}")
    
    def save(self):
        
        pass
    
    def predict(self, sample: list, headers: list):
    
        prediction = {c: -1 for c in self.models.keys()}
        
        for c, v in self.models.items():
            
            feature_idx = headers.index(self.feature_per_class[c])
            prediction[c] = 0 if v.model.hypothesis(sample[feature_idx]) < 0.5 else 1
            
        print(prediction)

def build_all_features(dataset):
    
    target_feature = "Hogwarts House"
    classes = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    features = ["Arithmancy",
                "Astronomy",
                "Herbology",
                "Defense Against the Dark Arts",
                "Divination",
                "Muggle Studies",
                "Ancient Runes",
                "History of Magic",
                "Transfiguration",
                "Potions",
                "Care of Magical Creatures",
                "Charms",
                "Flying"]
    models = {c : [GradientDescent(model=LogisticRegression,
                                           algorithm=MiniBatch,
                                           learning_rate=0.2,
                                           epochs=500,
                                           debug=False) for _ in range(len(features))]
                       for c in classes}

    row = 4
    col = 13
    fig, axs = pyplot.subplots(row, col, figsize=(40, 15))
    
    def normalize_x(data: list[tuple[float, float]]) -> list[tuple[float, float]]:

        x = [row[0] for row in data]

        x_mean =  Statistics.mean(x)
        x_std = Statistics.std(x)
        
        data = [((row[0] - x_mean) / x_std, row[1]) for row in data]
        
        return data

    for i, c in enumerate(classes):
        
        for j, feature in enumerate(features):
            
            m = models[c][j]
        
            print(f"Fitting class {c} - {feature}")
            
            # for each class, we build a logreg model -> C vs not C
            
            dataset.clean(target_feature=target_feature, include_features=[feature])
            
            encode = lambda target: 1 if target == c else 0
            
            data = dataset.get_cleaned_data()
            target = dataset.get_target()
            data = [(row[0], encode(target[i])) for i, row in enumerate(data)]
            data = normalize_x(data)
            
            # print(f"{c}: {data}")
            
            m.fit(data)
            print(f"weight: {m.get_weight()}")
            print(f"bias: {m.get_bias()}")
            
            x_range = np.linspace(min([row[0] for row in data]), max([row[0] for row in data]), 100)
            # Calculate the y-values using the model's hypothesis function
            y_range = [m.model.hypothesis(x) for x in x_range]
            
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].scatter([row[0] for row in data],
                            [row[1] for row in data])
            axs[i, j].plot(x_range, y_range, color='red')
            axs[i, j].set_title(f"{c} - {feature}", fontsize=8)

    pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                           wspace=0.5, hspace=0.5)
    pyplot.savefig("assets/logreg.png")

if __name__ == "__main__":
    
    ds = Dataset("datasets/dataset_train.csv")
    
    build_all_features(ds)