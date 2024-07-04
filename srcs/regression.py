import math
from typing import Callable
from abc import ABC, abstractmethod

class RegressionModel(ABC):
    
    """
    Regression for a single feature.
    """

    def __init__(self, init_weight: float = 1, init_bias: float = 1) -> None:
        
        self.w = init_weight
        self.b = init_bias

    def update(self, weight, bias):
        self.w = weight
        self.b = bias
    
    @abstractmethod
    def hypothesis(self, x):
        pass
    
    @abstractmethod
    def cost(self, samples: list[(float, float)]):
        pass
    
    @abstractmethod
    def dC_dw(self, x, y):
        pass
    
    @abstractmethod
    def dC_db(self, x, y):
        pass
    
    def __str__(self) -> str:
        return f"weight: {self.w}, bias: {self.b}"

class LogisticRegression(RegressionModel):
    
    """
    Logistic Regression class for a singular feature.
    
    """
    
    def hypothesis(self, x):
        
        # sigmoid function
        return 1 / (1 + math.e ** (- (self.b + (self.w * x))))

    def cost(self, samples: list[(float, float)]):
        
        total = 0
        
        count = len(samples)
        
        for x_actual, y_actual in samples:
            
            y_pred = self.hypothesis(x_actual)
            
            total += (y_actual * math.log(y_pred)) + ((1 - y_actual) * math.log(1 - y_pred))
        
        return - total / count
    
    def dC_dw(self, x, y):
        
        return (self.hypothesis(x) - y) * x
        
    def dC_db(self, x, y):
        
        return (self.hypothesis(x) - y)
