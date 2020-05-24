import numpy as np
import unittest

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
    
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            # memorize parent creator to output
            output.set_creator(self)
        # memorize inputs
        self.inputs = inputs
        # memorize outputs
        self.outputs = outputs
        return outputs
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, x):
        raise NotImplementedError()
        
        
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
        
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

