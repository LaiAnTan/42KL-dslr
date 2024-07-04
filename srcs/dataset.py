from tabulate import tabulate
from stats import Statistics

missing_count = 0

def try_float(value: str) -> float | str:
        
    try:
        value = float(value)
    except ValueError:
        pass
    
    return value

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

class Dataset:
    
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
    
    def __init__(self, path: str) -> None:
        
        self.path = path
        self.headers = []
        self.data = []
        self.target = []
        
        with open(path) as file:
    
            self.headers = next(file).strip().split(',')
            
            for line in file:
                row = line.strip().split(',')
                self.data.append(list(map(try_float, row))) # tries to convert all numeric values in the row to floats

    def get_feature(self, feature: str) -> list:
    
        """
        Extracts a feature from a row major dataset.
        """
        
        index = self.headers.index(feature)

        return [row[index] for row in self.data]
    
    def clean(self, target_feature, remove_features):
    
        """
        Function that cleans the dataset by removing unwanted features.
        Also handles missing values.
        """
        
        def calculate_imputation_values(target, target_idx, feature_idx, raw: list[list]):

            # median of the current feature with respect to the class it is in
            return Statistics.median([float(row[feature_idx]) for row in raw \
                if row[feature_idx] != '' and row[target_idx] == target])
        
        global missing_count

        ignore_idxs = set()
        
        target_idx = self.headers.index(target_feature)
        
        ignore_idxs.add(target_idx)
        
        for feature in remove_features:
            ignore_idxs.add(self.headers.index(feature))
        
        cleaned = []
        
        for row in self.data:
            
            cleaned_row = []
            self.target.append(row[target_idx])

            for i, val in enumerate(row):
                
                
                if i in ignore_idxs:
                    continue
                
                #! handle missing values here
                if val == '':
                    missing_count += 1
                    # median imputation for now
                    val = calculate_imputation_values(row[target_idx], target_idx, i, self.data)
                
                cleaned_row.append(val)
            
            cleaned.append(cleaned_row)
        
        cleaned_header = []
        
        for i in range(len(self.headers)):
            cleaned_header.append(self.headers[i]) if i not in ignore_idxs else None

        self.headers = cleaned_header
        self.data = cleaned

    def describe(self):
        
        described = []
        feature_data = []
        
        for feature in self.headers:        
            feature_data.append((feature, self.get_feature(feature)))
        
        for feature, data in feature_data:
            described.append([func(data) for func in self.stats.values()])
    
        row_count = len(self.stats.keys())
        
        rows = [[name] for name in self.stats.keys()]
        
        for val in described:
            for i in range(row_count):
                rows[i].append(val[i])
        
        print(tabulate(rows, headers=["Metric"] + self.headers, numalign="right"))