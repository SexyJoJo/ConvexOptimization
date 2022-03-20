import torch

def grad_descent(x, a, b, n, t=1):
    """
    梯度下降法
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

    def grad_sum(grad):
        """计算梯度向量的二范数"""
        abs_sum = 0
        for i in grad:
            abs_sum += pow(i, 2)
        return abs_sum

    grad = calculate_grad(x)
    while grad_sum(grad) > n:
        grad = calculate_grad(x)  # 梯度
        delta_x = -grad  # 梯度方向
        while f(x + t * delta_x) > f(x) + a * t * grad.t() @ delta_x:
            t = b * t
        x = x + t * delta_x
        print(x)
    return x


if __name__ == '__main__':
    x = torch.tensor([-2., 4.], requires_grad=True)
    best = grad_descent(x, a=0.2, b=0.5, n=0.0001)




