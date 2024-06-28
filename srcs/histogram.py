import sys
import os
from matplotlib import pyplot

from describe import extract, clean

def filter_feature_by_category(data: dict[str, list[int | float]],
                               target: list[int],
                               target_labels: dict[str, int],
                               feature: str,
                               category: str,
                               ):
    
    feature_data = data[feature]
    
    encoded_target = target_labels[category]
    
    ret = []
    
    for t, v in zip(target, feature_data):
        
        if t == encoded_target:
            ret.append(v)

    return ret

def main():
    
    if len(sys.argv) != 2:
        return print("Wrong number of arguments")
    
    if not os.path.isfile(sys.argv[-1]) or sys.argv[-1].split('.')[-1] != 'csv':
        return print(f"Invalid file")

    extracted = extract(sys.argv[-1])
    
    target, target_labels, cleaned = clean(extracted)
    
    l = filter_feature_by_category(cleaned, target, target_labels, "Astronomy", "Ravenclaw")
    g = filter_feature_by_category(cleaned, target, target_labels, "Astronomy", "Hufflepuff")

if __name__ == "__main__":
    
    main()