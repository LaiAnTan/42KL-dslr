
from dataset import Dataset
from ovr import OneVsRestClassifier

def main():
    
    ds_train = Dataset("datasets/dataset_train.csv")
    ds_test = Dataset("datasets/dataset_test.csv")
    feature_per_class = {
        "Gryffindor": "Flying", # "History of Magic", "Transfiguration"
        "Hufflepuff": "Ancient Runes",
        "Ravenclaw": "Charms", # "Muggle Studies"
        "Slytherin": "Divination" 
    }
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
    
    ovr = OneVsRestClassifier(ds_train, "Hogwarts House", feature_per_class)
    
    ovr.fit_models()
    
    ds_test.clean("Hogwarts House", features)
    test_data = ds_test.get_cleaned_data()
    
    # ovr.save()
    for i, sample in enumerate(test_data):
        print(f"{i},{ovr.predict(sample, ds_test.get_cleaned_headers())}")
    

if __name__ == "__main__":
    
    main()