import sys
import os
from tabulate import tabulate
from typing import Callable, Optional

from stats import Statistics

missing_count = 0

def label_encoder(data: list[str]) -> list[int]:
    
    """
    Unused for now
    """
    
    unique = list(set(data))
    
    labels = {}
    
    for i, label in enumerate(unique):
        
        labels[label] = i
    
    for i, entry in enumerate(data):
        
        data[i] = labels[entry]
    
    return labels, data

def get_feature(headers: list[str], data: list[list], feature: str) -> list:
    
    """
    Extracts a feature from a row major dataset.
    """
    
    index = headers.index(feature)

    return [row[index] for row in data]

def try_float(value: str) -> float | str:
        
    try:
        value = float(value)
    except ValueError:
        pass
    
    return value

def extract(path: str) -> tuple[list, list[list]]:
    
    raw = []
    
    with open(path) as file:
    
        headers = next(file).strip().split(',')
        
        for line in file:
            row = line.strip().split(',')
            raw.append(list(map(try_float, row))) # applies handler on all values in a row
    
    return headers, raw

def calculate_imputation_values(target, target_idx, feature_idx, raw: list[list]):

    # median of the current feature with respect to the class it is in
    return Statistics.median([float(row[feature_idx]) for row in raw if row[feature_idx] != '' and row[target_idx] == target])

def clean(headers: list[str], raw: list[list]):
    
    """
    Function that cleans the dataset by removing unwanted features.
    Also handles missing values.
    """
    
    global missing_count

    target_feature = "Hogwarts House"
    remove_features = ["Index", "First Name", "Last Name", "Birthday", "Best Hand"]

    ignore_idxs = set()
    
    target_idx = headers.index(target_feature)
    
    ignore_idxs.add(target_idx)
    
    for feature in remove_features:
        ignore_idxs.add(headers.index(feature))
    
    cleaned = []
    
    for row in raw:
        
        cleaned_row = []

        for i, val in enumerate(row):
            
            if i in ignore_idxs:
                continue
            
            #! handle missing values here
            if val == '':
                missing_count += 1
                # median imputation for now
                val = calculate_imputation_values(row[target_idx], target_idx, i, raw) 
            
            cleaned_row.append(val)
        
        cleaned.append(cleaned_row)
    
    cleaned_header = []
    
    for i in range(len(headers)):
        cleaned_header.append(headers[i]) if i not in ignore_idxs else None

    return cleaned_header, cleaned

def describe(headers: list[str], cleaned: list[list]):
    
    described = []
    stats = {
            "Count": Statistics.count,
            "Mean": Statistics.mean,
            "Var": Statistics.var,
            "Std": Statistics.std,
            "Min": min,
            "Max": max,
            "Range": Statistics.range,
            "25%": Statistics.lower_quartile,
            "50%": Statistics.median,
            "75%": Statistics.upper_quartile,
            "IQR": Statistics.interquartile_range,
            "Skewness": Statistics.skewness
        }
    
    feature_data = []
    
    for feature in headers:        
        feature_data.append((feature, get_feature(headers, cleaned, feature)))
    
    for feature, data in feature_data:
        
        described.append([func(data) for func in stats.values()])

    return stats, described

def main():
    
    if len(sys.argv) != 2:
        return print("Wrong number of arguments")
    
    if not os.path.isfile(sys.argv[-1]) or sys.argv[-1].split('.')[-1] != 'csv':
        return print(f"Invalid file")

    headers, extracted = extract(sys.argv[-1])

    cleaned_headers, cleaned = clean(headers, extracted)
    
    stats, described = describe(cleaned_headers, cleaned)
    
    headers = ["Metric"] + cleaned_headers
    
    row_count = len(stats.keys())
    
    rows = [[name] for name in stats.keys()]
    
    for val in described:
        
        for i in range(row_count):
            rows[i].append(val[i])
    
    print(tabulate(rows, headers=headers, numalign="right"))

if __name__ == "__main__":
    
    main()

"""

train dataset

numerical category count: 12
total entries: 1600
total data points: 19200
missing data points: 386

percentage data loss: 2.01%

"""

