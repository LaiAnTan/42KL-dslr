import sys

from dataset import Dataset
from ovr import OneVsRestClassifier

def main():
    
    ds_train = Dataset(sys.argv[-1])

    feature_per_class = {
        "Gryffindor": "Flying", # "History of Magic", "Transfiguration"
        "Hufflepuff": "Ancient Runes",
        "Ravenclaw": "Charms", # "Muggle Studies"
        "Slytherin": "Divination" 
    }
    
    target_feature = "Hogwarts House"
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
    
    ds_train.clean(target_feature, features)
    
    ovr = OneVsRestClassifier(ds_train, "Hogwarts House", feature_per_class)
    
    ovr.fit_models()
    
    ovr.save("models/hogwarts_logreg_ovr.json")

if __name__ == "__main__":
    
    main()