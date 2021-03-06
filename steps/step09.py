import numpy as np

class Variable:
    
    def __init__(self, data):
        
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        
        while funcs:
            f = funcs.pop() # 関数を取得
            x, y = f.input, f.output # 関数の入出力を取得
            x.grad = f.backward(y.grad)  # 関数のbackwardメソッドを呼ぶ
            
            if x.creator is not None:
                funcs.append(x.creator) # 1つ前の関数をリストに追加

class Function:
    
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        # memorize parent creator to output
        output.set_creator(self)
        # memorize input value
        self.input = input
        # memorize output
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, x):
        raise NotImplementedError()
        
class Square(Function):
    
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = gy * x * 2
        return gx
    
class Exp(Function):
    
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        # exp(x)の微分はexp(x)
        gx = np.exp(x) * gy
        return gx
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
