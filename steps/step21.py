import numpy as np
import weakref
import contextlib

class Variable:
    __array_priority__ = 200
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name
        
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return f'variable( {p} )'
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(self, other)
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(self, other)
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def cleargrad(self):
        self.grad = None
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
        
    def backward(self, retrain_grad=False):
        
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop() # 関数を取得
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retrain_grad:
                for y in f.outputs:
                    y().grad = None # yはweakref

class Function:
    # *inputsで可変長引数にする
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # アスタリスクをつけてUnpacking
        if not isinstance(ys, tuple): # tupleでない場合の追加対応
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            # generationを設定
            self.generation = max([x.generation for x in inputs])
        
            for output in outputs:
                # memorize parent creator to output
                output.set_creator(self)
            # memorize inputs
            self.inputs = inputs
            # memorize outputs
            self.outputs = [weakref.ref(output) for output in outputs]
        
        # リストの要素が1つの時は最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, x):
        raise NotImplementedError()
        
class Config:
    enable_backprop = True
        
        
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

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

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
        
def no_grad():
    return using_config('enable_backprop', False)
