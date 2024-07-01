from typing import Callable
from abc import ABC, abstractmethod
from regression import RegressionModel

class GradientDescentAlgorithm(ABC):
    
    def __init__(self, learning_rate: float, epochs: int,
                 reg_model: RegressionModel,
                 debug=False
                 ) -> None:
        
        self.lr = learning_rate
        self.epochs = epochs
        self.reg_model = reg_model
        self.debug = debug
        self.w = None
        self.b = None
    
    @abstractmethod
    def run(self, data: list[float, float]):
        pass


class Batch(GradientDescentAlgorithm):
    
    def run(self, data: list[tuple[float, float]]):
        
        sample_count = len(data)

        for step in range(1, self.epochs + 1):

            w_grad = 0  # d Cost / dw
            b_grad = 0  # d Cost / db

            for x, y in data:
                w_grad += self.reg_model.dC_dw(x, y)
                b_grad += self.reg_model.dC_db(x, y)
            
            w_grad /= sample_count
            b_grad /= sample_count

            self.w -= self.learning_rate * w_grad
            self.b -= self.learning_rate * b_grad

            if self.debug:
                cost = self.reg_model.cost(data)
                print(f"Epoch: {step} - Cost: {cost}")
            
            self.reg_model.update(self.w, self.b)
        
        print("Complete")

class MiniBatch(GradientDescentAlgorithm):

    def __init__(self, learning_rate: float, epochs: int, reg_model: RegressionModel, debug=False, batch_size: int = 32) -> None:
        super().__init__(learning_rate, epochs, reg_model, debug)
        
        self.batch_size = batch_size
        
    def run(self, data: list[tuple[float, float]]):
        
        for step in range(1, self.epochs + 1):
            
            for i in range(0, len(data), self.batch_size):
                
                batch = data[i: i + self.batch_size]
                curr_batch_size = len(batch)
                
                w_grad = 0
                b_grad = 0
                
                for x, y in batch:
                    
                    w_grad += self.reg_model.dC_dw(x, y)
                    b_grad += self.reg_model.dC_db(x, y)
                
                w_grad /= curr_batch_size
                b_grad /= curr_batch_size
            
                self.w -= self.learning_rate * w_grad
                self.b -= self.learning_rate * b_grad

                self.reg_model.update(self.w, self.b)

            if self.debug:
                cost = self.reg_model.cost(data)
                print(f"Epoch: {step} - Cost: {cost}")
            
        print("Complete")

class Stochastic(GradientDescentAlgorithm):
    
    """
    Gradient Descent that updates w and b for each sample.
    """

    def run(self, data: list[tuple[float, float]]):

        for step in range(1, self.epochs + 1):
            
            w_grad = 0  # d Cost / dw
            b_grad = 0  # d Cost / db

            for x, y in data:
                w_grad += self.reg_model.dC_dw(x, y)
                b_grad += self.reg_model.dC_db(x, y)

                self.w -= self.learning_rate * w_grad
                self.b -= self.learning_rate * b_grad
                
                self.reg_model.update(self.w, self.b)

            if self.debug:
                cost = self.reg_model.cost(data)
                print(f"Epoch: {step} - Cost: {cost}")
            
        print("Complete")

class GradientDescent:
    
    """
    Gradient Descent Runner Class w/ 1 feature.
    """

    def __init__(self, model: RegressionModel,
                 algorithm: GradientDescentAlgorithm,
                 learning_rate: float = 0.1,
                 epochs: float = 100,
                 init_weight: float = 1,
                 init_bias: float = 1,
                 debug: bool = False) -> None:
        
        
        self.model = model(init_weight, init_bias)
        self.algorithm = algorithm(learning_rate, epochs, model, debug)
    
    def solve(self, data: list[tuple[float, float]]):
        
        self.algorithm.run(data)

        return self.algorithm.w, self.algorithm.b
    
    def save(self):
        pass
