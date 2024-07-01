
from gd import GradientDescent, Batch, MiniBatch, Stochastic
from regression import LogisticRegression

def label_encoder(data: list[str]) -> list[int]:
    
    """
    Encode labels with numerical values.
    """
    
    unique = list(set(data))
    
    labels = {}
    
    for i, label in enumerate(unique):
        
        labels[label] = i
    
    for i, entry in enumerate(data):
        
        data[i] = labels[entry]
    
    return labels, data

def main():
    
    data = []
    
    model = LogisticRegression
    algo = MiniBatch
    lr = 0.1
    epochs = 100
    debug = True
    
    gd = GradientDescent(model, algo, lr, epochs, debug);

    w, b = gd.solve(data)
    
    print(w, b)

if __name__ == "__main__":
    
    main()