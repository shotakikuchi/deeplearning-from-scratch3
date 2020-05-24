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
        
    def cleargrad(self):
        self.grad = None
        
    def backward(self):
        
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        
        while funcs:
            f = funcs.pop() # 関数を取得
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator) # 1つ前の関数をリストに追加

class Function:
    # *inputsで可変長引数にする
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # アスタリスクをつけてUnpacking
        if not isinstance(ys, tuple): # tupleでない場合の追加対応
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            # memorize parent creator to output
            output.set_creator(self)
        # memorize inputs
        self.inputs = inputs
        # memorize outputs
        self.outputs = outputs
        
        # リストの要素が1つの時は最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, x):
        raise NotImplementedError()
        
        
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
def square(x):
    return Square()(x)
        
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

