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
        "Skewness": Statistics.skewness,
        "Kurtosis": Statistics.kurtosis
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
    
    def clean(self, target_feature, include_features):
    
        """
        Function that cleans the dataset by removing unwanted features.
        Also handles missing values by imputation.
        """
        
        global missing_count

        cleaned_data = [[] for _ in range(len(self.data))]
        cleaned_headers = []
        
        self.target = self.get_feature(target_feature)
        
        for i, header in enumerate(self.headers):
            
            if header == target_feature or header not in include_features:
                continue
            
            cleaned_headers.append(header)
            imputate_val = Statistics.median(list(filter(lambda x: x != '', self.get_feature(header))))
            
            for j, row in enumerate(self.data):

                cleaned_data[j].append(row[i] if row[i] != '' else imputate_val)

        self.data = cleaned_data
        self.headers = cleaned_headers

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
    
    def get_data(self):
        return self.data
    
    def get_headers(self):
        return self.headers
    
    def get_target(self):
        return self.target
