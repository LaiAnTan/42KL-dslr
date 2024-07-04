import sys

from dataset import Dataset
from ovr import OneVsRestClassifier

def calculate_accuracy():
    
    correct_count = 0
    total_count = 0
    
    with open('datasets/dataset_prediction.csv', 'r') as prediction_file, \
        open('datasets/dataset_truth.csv', 'r') as truth_file:
        
        next(prediction_file)
        next(truth_file)
        
        for i, (prediction, truth) in enumerate(zip(prediction_file, truth_file)):
            
            prediction = prediction.strip().split(',')[1]
            truth = truth.strip().split(',')[1]
            
            if prediction == truth:
                correct_count += 1
            
            else:
                print(f"{i}: {prediction} is not {truth}")
            
            total_count += 1

        print(f"Accuracy: {correct_count / total_count * 100}%")

def main():
    
    ds_test = Dataset("datasets/dataset_test.csv")
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
    
    ds_test.clean("Hogwarts House", features)
    test_data = ds_test.get_data()
    
    ovr = OneVsRestClassifier()
    ovr.load(sys.argv[-1])
    
    with open('datasets/dataset_prediction.csv', 'w') as file:
        
        file.write("Index,Hogwarts House\n")
    
        for i, sample in enumerate(test_data):
            
            if i == len(test_data) - 1:
                file.write(f"{i},{ovr.predict(sample, ds_test.get_headers())}")
                break
            
            file.write(f"{i},{ovr.predict(sample, ds_test.get_headers())}\n")
    
    calculate_accuracy()

if __name__ == "__main__":
    
    main()