from math import sqrt

def is_even(num: int) -> bool:
    return num % 2 == 0

class Statistics:
    
    @classmethod
    def count(cls, data: list) -> int:
        return len(data)
    
    @classmethod
    def range(cls, data: list) -> float:
        return max(data) - min(data)

    @classmethod
    def mean(cls, data: list) -> float:
        return sum(data) / len(data)

    @classmethod
    def median(cls, data: list) -> float:

        count = len(data)
        sorted_data = sorted(data)
        middle = count // 2
        
        if is_even(count):
            return (sorted_data[middle] + sorted_data[middle- 1]) / 2

        return sorted_data[middle]

    @classmethod
    def mode(cls, data: list) -> int:
        
        counts = {}
        
        for val in data:
            
            if val in counts.keys():
                counts[val] += 1
            else:
                counts[val] = 0
        
        return max(counts, key=counts.get)

    @classmethod
    def lower_quartile(cls, data: list) -> float:
        
        sorted_data = sorted(data)
        count = len(data)
        middle = count // 2
        
        if is_even(count):
            lower_data = sorted_data[:middle]
        else:
            lower_data = sorted_data[0 : middle + 1]
        
        lower_count = len(lower_data)
        lower_middle = lower_count // 2
    
        if is_even(lower_count):
            return (lower_data[lower_middle] + lower_data[lower_middle - 1]) / 2
            
        return lower_data[lower_middle]

    @classmethod
    def upper_quartile(cls, data: list) -> float:

        sorted_data = sorted(data)
        count = len(data)
        middle = count // 2
        
        if is_even(count):
            upper_data = sorted_data[middle:]
        else:
            upper_data = sorted_data[middle + 1:]
        
        upper_count = len(upper_data)
        upper_middle = upper_count // 2
    
        if is_even(upper_count):
            return (upper_data[upper_middle] + upper_data[upper_middle - 1]) / 2
            
        return upper_data[upper_middle]
    
    @classmethod
    def interquartile_range(cls, data: list) -> float:
        return cls.upper_quartile(data) - cls.lower_quartile(data)

    @classmethod
    def var(cls, data: list) -> float:
        
        avg = cls.mean(data)
        
        # sample variance
        return sum([(val - avg) ** 2 for val in data]) / (len(data) - 1)

    @classmethod
    def std(cls, data: list) -> float:
        return sqrt(cls.var(data))
    
    @classmethod
    def skewness(cls, data: list) -> float:
        
        mean = cls.mean(data)
        std = cls.std(data)
        n = len(data)
        
        # sample skewness
        return (n / ((n - 1) * (n - 2))) * \
            (sum([(val - mean) ** 3 for val in data])) / (std ** 3)
    
    @classmethod
    def kurtosis(cls, data: list) -> float:
        
        mean = cls.mean(data)
        std = cls.std(data)
        n = len(data)
        
        # sample kurtosis
        return (((n * (n - 1)) / ((n - 1) * (n - 2) * (n - 3))) * \
            (sum([(val - mean) ** 4 for val in data])) / (std ** 4)) - \
                ((3 * (n - 1) ** 2) / ((n - 1) * (n - 2)))

if __name__ == "__main__":
    
    l = [1, 2, 5, 4, 2, 3, 1, 2, 1, 1, 5, 2, 3, 5, 5, 5, 5, 5, 5, 3, 2, 1, 2, 5, 5, 5, 5, 5]
    
    print(f"Mode: {Statistics.mode(l)}")

