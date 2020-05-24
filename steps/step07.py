import numpy as np

class Variable:
    
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        f = self.creator # 1. 関数を取得
        if f is not None:
            x = f.input # 2. 関数の入力を取得
            x.grad = f.backward(self.grad) # 3. 関数のbackwardメソッドを呼ぶ
            x.backward() # 自分より1つ前の変数のbackwardメソッドを呼ぶ（再帰）
        
class Function:
    
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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