import torch

def newton_descent(x, a, b, n, t=1):
    """
    等式约束牛顿法
    :param x: 初始解
    :param a: 属于(0, 0.5)
    :param b: 属于(0, 1)
    :param t: 初始化为1
    :param n: 用于终止条件误差判断
    :return: 最优解
    """
    def f(x):
        """通过标准二次型计算目标值"""
        b = torch.tensor([-2., 0.]).unsqueeze(1)
        A = torch.tensor([[3., -1.], [-1., 1.]])
        return 1/2*x.t()@A@x+b.t()@x

    def calculate_grad(x):
        """计算梯度"""
        y = f(x)
        return torch.autograd.grad(y, x, retain_graph=True, create_graph=True)[0]

    def calculate_Hessian(x):
        y = torch.tensor([])
        for anygrad in calculate_grad(x):
            temp = torch.autograd.grad(anygrad, x, retain_graph=True)[0]
            y = torch.cat((y, temp))    # 1表示按列拼接
        return y.view(x.size()[0], -1)

    def newton_reduction(grad, hessian):
        return grad.t() @ hessian.inverse() @ grad

    def newton_footpath(grad, hessian):
        return -1 * hessian.inverse() @ grad

    grad = calculate_grad(x)
    hessian = calculate_Hessian(x)

    while newton_reduction(grad, hessian) > n:
        grad = calculate_grad(x)
        hessian = calculate_Hessian(x)
        delta_x = newton_footpath(grad, hessian)
        reduction = newton_reduction(grad, hessian)
        while f(x + t * delta_x) > f(x) + a * t * reduction:
            t = b * t
        x = x + t * delta_x
        print(x)
    return x


if __name__ == '__main__':
    x = torch.tensor([-2., 4.], requires_grad=True)
    best = newton_descent(x, a=0.2, b=0.5, n=0.0001)
