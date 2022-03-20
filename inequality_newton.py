import torch
import math

def inequality_newton(x, s, equation_coefficient, equation_num, alpha, beta, n, t=1, error=0.00001):
    """
    等式约束牛顿法
    :param error: 允许的误差值
    :param equation_coefficient: 等式约束系数向量
    :param x: 初始解
    :param alpha: 属于(0, 0.5)
    :param beta: 属于(0, 1)
    :param t: 初始化为1
    :param n: 用于终止条件误差判断
    :return: 最优解
    """

    def f(x, s):
        """通过标准二次型计算目标值"""
        # b = torch.tensor([-2., 0.]).unsqueeze(1)
        # A = torch.tensor([[3., -1.], [-1., 1.]])
        # return 1 / 2 * x.t() @ A @ x + b.t() @ x
        x1 = x[0]
        x2 = x[1]
        return 3/2 * (x1 ** 2) + 1/2 * (x2 ** 2) + x1 * x2 - torch.log(s[0] - 2 * x1 - x2)

    def calculate_grad(need_grad):
        """计算梯度"""
        y = f(x, s)
        return torch.autograd.grad(y, need_grad, retain_graph=True, create_graph=True)[0]

    def calculate_Hessian(x):
        y = torch.tensor([])
        for anygrad in calculate_grad(x):
            temp = torch.autograd.grad(anygrad, x, retain_graph=True)[0]
            y = torch.cat((y, temp))  # 1表示按列拼接
        return y.view(x.size()[0], -1)

    z = torch.cat((x, s), dim=0)
    while True:
        grad = calculate_grad(z)
        hessian = calculate_Hessian(z)

        # 拼接矩阵
        temp1 = torch.cat((hessian, equation_coefficient.t()), dim=1)
        temp2 = torch.cat((equation_coefficient, torch.tensor([[0]])), dim=1)
        cat_matrix = torch.cat((temp1, temp2), dim=0)

        # 计算delta_x和delta_v
        delta = -torch.cat((grad, (equation_coefficient @ x - equation_num).squeeze(1)), dim=0) @ cat_matrix.inverse()
        delta_x = delta[:2]
        delta_v = delta[2]
        print("delta_x,v:  ", delta_x, delta_v)

        # 计算r和delta用于回溯直线搜索
        r = torch.cat((grad + equation_coefficient.t()@v, (equation_coefficient @ x - equation_num).squeeze(1)), dim=0)
        grad2 = calculate_grad(x + t * delta_x)
        delta_r = torch.cat((grad2 + equation_coefficient.t() @ (v + t * delta_v),
                             (equation_coefficient @ (x + t * delta_x) - equation_num).squeeze(1)), dim=0)

        # 回溯直线搜索
        while torch.norm(delta_r, p="fro") >= (1 - alpha * t) * torch.norm(r, p="fro"):
            print("norm:", torch.norm(delta_r, p="fro"), (1 - alpha * t) * torch.norm(r, p="fro"))
            t = beta * t

        x = x + t * delta_x
        v = v + t * delta_v
        if equation_coefficient @ x - equation_num.squeeze(1) < error and torch.norm(delta_r, p='fro') < n:
            break
        print("________________________")

    return x, v


if __name__ == '__main__':
    x = torch.tensor([0., 1.], requires_grad=True)
    equation_coefficient = torch.tensor([[0., 0., 1.]])
    equation_num = torch.tensor([[0.]])
    s = torch.tensor([2.])
    best_x, best_v = inequality_newton(x, s=s, alpha=0.2, beta=0.5, n=0.0001,
                                       equation_coefficient=equation_coefficient, equation_num=equation_num)
    print("best:  ", best_x, best_v)
