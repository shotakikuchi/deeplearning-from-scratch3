from steps.step01 import Variable

# 微分 中心差分方
def numerical_diff(f,x, eps=1e-4):
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data-y0.data)/(2*eps)