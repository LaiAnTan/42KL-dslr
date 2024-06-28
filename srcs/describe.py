import sys
import os
from tabulate import tabulate

from stats import Statistics

def label_encoder(data: list[str]) -> list[int]:
    
    unique = list(set(data))
    
    labels = {}
    
    for i, label in enumerate(unique):
        
        labels[label] = i
    
    for i, entry in enumerate(data):
        
        data[i] = labels[entry]
    
    return labels, data

def extract(path: str) -> dict[str, list[int | float]]:
    
    extracted = {}
    
    with open(path) as file:
    
        features = next(file).strip().split(',')
        
        for feature in features:
            extracted[feature] = []
        
        for line in file:
    
            row = line.strip().split(',')
            
            # !! handle this better later rather than skipping the row!
            if '' in row:
                continue
    
    
            # ?? do not know if i should categorise by row or by feature
            for feature, val in zip(features, row):
                
                if val == '':
                    val = None # change this to some value to handle missing data
                
                try:
                    val = float(val)
                except ValueError:
                    pass
                    
                extracted[feature].append(val)
    
    return extracted

def clean(raw: dict[str, list[int | float]]):
    
    """
    Function that cleans the dataset by:
        - removing unwanted features
        - extracting target feature and encoding it
        
    :return: 
        1. target list that contains all encoded targets with the order preserved
        2. target_labels dictionary which acts as a mapping from the original to encoded target labels
        3. and also the data after features have been removed with order preserved
    """
    
    target_feature = "Hogwarts House"
    remove_features = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
    
    # extract target feature
    if target_feature in raw.keys():
        target = raw.get(target_feature)

    # remove useless features
    for feature in remove_features:
        del raw[feature]
    
    # encode labels
    target_labels, target = label_encoder(target)
    
    return target, target_labels, raw


def describe(cleaned: dict[str, list[int | float]]):
    
    s = Statistics()
    
    described = {}
    stats = {
            "Count": s.count,
            "Mean": s.mean,
            "Var": s.var,
            "Std": s.std,
            "Min": min,
            "Max": max,
            "Range": s.range,
            "25%": s.lower_quartile,
            "50%": s.median,
            "75%": s.upper_quartile,
            "IQR": s.interquartile_range,
            "Skewness": s.skewness
        }
    
    for feature, data in cleaned.items():
        
        described[feature] = [func(data) for func in stats.values()]

    return stats, described

def main():
    
    if len(sys.argv) != 2:
        return print("Wrong number of arguments")
    
    if not os.path.isfile(sys.argv[-1]) or sys.argv[-1].split('.')[-1] != 'csv':
        return print(f"Invalid file")

    extracted = extract(sys.argv[-1])
    
    target, target_labels, cleaned = clean(extracted)
    
    stats, described = describe(cleaned)
    
    headers = ["Metric"] + list(described.keys())
    
    row_count = len(stats.keys())
    
    rows = [[name] for name in stats.keys()]
    
    for val in described.values():
        
        for i in range(row_count):
            rows[i].append(val[i])
    
    print(tabulate(rows, headers=headers, numalign="right"))

if __name__ == "__main__":
    
    main()

