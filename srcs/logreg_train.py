
from dataset import Dataset
from gd import GradientDescent, Batch, MiniBatch, Stochastic
from regression import LogisticRegression

def main():
    
    target_feature = "Hogwarts House"
    remove_features = ["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
    
    ds = Dataset("datasets/dataset_train.csv")
    ds.clean(target_feature, remove_features)
    
    model = LogisticRegression
    algo = MiniBatch
    lr = 0.1
    epochs = 100
    debug = True
    
    gd = GradientDescent(model, algo, lr, epochs, debug);
    
    data = []

    w, b = gd.fit(data)
    
    print(w, b)

if __name__ == "__main__":
    
    main()