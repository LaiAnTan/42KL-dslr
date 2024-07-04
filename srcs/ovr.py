from matplotlib import pyplot
import random
import json

from dataset import Dataset
from gd import GradientDescent, Batch, MiniBatch, Stochastic
from regression import LogisticRegression
from stats import Statistics

class OneVsRestClassifier:
    
    MODEL = LogisticRegression
    ALGORITHM = MiniBatch
    LEARNING_RATE = 0.1
    EPOCHS = 200
    DEBUG = True
    SEED = 42
    
    def __init__(self, dataset: Dataset = None,
                 target: str = None,
                 feature_per_class: dict[str, str] = None) -> None:
        
        random.seed(self.SEED)
    
        if dataset is not None and \
            target is not None and \
            feature_per_class is not None:
    
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
            self.coefs = {c : None for c in self.feature_per_class.keys()}
            self.target = target

    def fit_models(self):
        
        def normalize_x(data: list[tuple[float, float]]) -> list[tuple[float, float]]:

            x = [row[0] for row in data]

            x_mean =  Statistics.mean(x)
            x_std = Statistics.std(x)
            
            data = [((row[0] - x_mean) / x_std, row[1]) for row in data]
            
            return data, x_mean, x_std

        for c in self.feature_per_class.keys():
            
            print(f"Fitting class {c}: {self.feature_per_class[c]}")
            
            # for each class, we build a logreg model -> C vs not C
            encode = lambda target: 1 if target == c else 0
            
            data = list(zip(self.dataset.get_feature(self.feature_per_class[c]),
                            map(encode, self.dataset.get_target())))
            data, mean, std = normalize_x(data)
            
            self.models[c].fit(data)
            print(f"weight: {self.models[c].get_weight()}")
            print(f"bias: {self.models[c].get_bias()}")
            self.coefs[c] = {
                "feature": self.feature_per_class[c],
                "weight": self.models[c].get_weight(),
                "bias": self.models[c].get_bias(),
                "mean": mean,
                "std": std
            }
    
    def save(self, path: str):
        
        with open(path, 'w') as file:
            json.dump(self.coefs, file)
    
    def load(self, path: str):
        
        with open(path, 'r') as file:
            self.coefs = json.load(file)
    
    def predict(self, sample: list, headers: list):
        
        prediction = {c: -1 for c in self.coefs.keys()}
        
        for c in self.coefs.keys():
            
            model = self.MODEL(self.coefs[c]["weight"], self.coefs[c]["bias"])
            
            feature_idx = headers.index(self.coefs[c]["feature"])
            prediction[c] = model.hypothesis((sample[feature_idx] - self.coefs[c]["mean"]) / self.coefs[c]["std"])
        
        return max(prediction, key=prediction.get)

def build_all_features(dataset: Dataset, features: list[str]):
    
    classes = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    models = {c : [GradientDescent(model=LogisticRegression,
                                           algorithm=MiniBatch,
                                           learning_rate=0.2,
                                           epochs=500,
                                           debug=False) for _ in range(len(features))]
                       for c in classes}

    row = 4
    col = len(features)
    fig, axs = pyplot.subplots(row, col, figsize=(120, 40))
    
    def normalize_x(data: list[tuple[float, float]]) -> list[tuple[float, float]]:

        x = [row[0] for row in data]

        x_mean =  Statistics.mean(x)
        x_std = Statistics.std(x)
        
        data = [((row[0] - x_mean) / x_std, row[1]) for row in data]
        
        return data
    
    def linspace(start, stop, num=50):
        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]

    for i, c in enumerate(classes):
        
        for j, feature in enumerate(features):
            
            m = models[c][j]
        
            print(f"Fitting class {c} - {feature}")
            
            # for each class, we build a logreg model -> C vs not C
            encode = lambda target: 1 if target == c else 0
            
            data = dataset.get_feature(feature)
            target = dataset.get_target()
            data = [(val, encode(target[i])) for i, val in enumerate(data)]
            data = normalize_x(data)
            
            m.fit(data)
            print(f"weight: {m.get_weight()}")
            print(f"bias: {m.get_bias()}")
            
            axs[i, j].grid(True)
            axs[i, j].scatter([row[0] for row in data],
                            [row[1] for row in data], alpha=0.5)
            x_values = [row[0] for row in data]
            axs[i, j].plot(x_range := linspace(min(x_values), max(x_values), 100),
                           [m.model.hypothesis(x) for x in x_range], color='red')
            axs[i, j].set_title(f"{c} - {feature}", fontsize=8)

    pyplot.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                           wspace=0.5, hspace=0.5)
    pyplot.savefig("assets/logreg.png")

if __name__ == "__main__":

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
    
    ds = Dataset("datasets/dataset_train.csv")

    ds.clean(target_feature, features)
    
    build_all_features(ds, features)