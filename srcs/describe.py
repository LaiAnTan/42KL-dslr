import sys
import os

from dataset import Dataset

def main():
    
    if len(sys.argv) != 2:
        return print("Wrong number of arguments")
    
    if not os.path.isfile(sys.argv[-1]) or sys.argv[-1].split('.')[-1] != 'csv':
        return print(f"Invalid file")
    
    target_feature = "Hogwarts House"
    include_features = ["Arithmancy",
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
    
    ds = Dataset(sys.argv[-1])
    ds.clean(target_feature, include_features)
    ds.describe()

if __name__ == "__main__":
    
    """

    train dataset

    numerical category count: 12
    total entries: 1600
    total data points: 19200
    missing data points: 386

    percentage data loss: 2.01%

    """
    
    main()

