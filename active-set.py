import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath


class ASM():
    def __init__(self, H, C, A, B):
        self.H = H
        self.C = C
        self.A = A
        self.B = B
        self.x = None
        self.WorkingSet = None

        self.path = []

    def derivative(self, x):
        de = self.H.dot(x) + self.C.T
        return de[0]

    def KKT(self, H, C, A, B):
        n = self.H.shape[0]
        wsc = len(self.WorkingSet)
        kkt_A = np.zeros((n + wsc, n + wsc))
        kkt_B = np.zeros((n + wsc))

        kkt_A[:n, :n] = H
        kkt_A[:n, n:] = -A.T
        kkt_A[n:, :n] = -A

        kkt_B[:n] = -C
        kkt_B[n:] = B[:, 0]

        return np.linalg.inv(kkt_A).dot(kkt_B)

    def Alpha(self, x, p):
        min_alpha = 1
        new_constraint = -1
        for i in range(self.A.shape[0]):
            if i in self.WorkingSet:
                continue
            else:
                bi = self.B[i]
                ai = self.A[i]
                atp = ai.dot(p)
                if atp >= 0:
                    continue
                else:
                    alpha = (bi - ai.dot(x)) / atp
                    if alpha < min_alpha:
                        min_alpha = alpha
                        new_constraint = i
        return min_alpha, new_constraint

    def solve(self):
        # 构建初始工作集, 默认第一个约束
        # 初始化当前点, 当前点为一约束下的任意有效解
        self.WorkingSet = [0]
        index = np.where(self.A[0] != 0)[0][0]
        value = self.A[0, index]
        t = self.B[0, 0] / value

        # 构造初始点
        self.x = np.zeros((self.A.shape[1]))
        self.x[index] = t
        count = self.H.shape[1]

        ### 博文作者的初始设置
        self.WorkingSet = [2, 4]
        self.x = np.array([2.0, 0.0])
        ####

        # 2. 循环
        maxtime = 100
        for _ in range(maxtime):
            self.path.append([self.x[0], self.x[1]])
            # 子命题参数
            c = self.derivative(self.x)
            a = self.A[self.WorkingSet]
            b = np.zeros_like(self.B[self.WorkingSet])

            dlambda = self.KKT(self.H, c, a, b)
            _lambda = dlambda[count:]
            d = dlambda[0:count]

            if np.linalg.norm(d, ord=1) < 1e-6:
                if _lambda.min() > 0:
                    break
                else:
                    if _lambda.shape[0] != 0:
                        index = np.argmin(_lambda)
                        del self.WorkingSet[index]
                        self.WorkingSet.sort()
            else:
                alpha, new_constraint = self.Alpha(self.x, d)
                self.x += alpha * d
                if alpha < 1:
                    self.WorkingSet.append(new_constraint)
                    self.WorkingSet.sort()


def main():
    # f(x) = (x0-1)^2 + (x1-2.5)^2
    # x1 - 2*x2 + 2 >= 0
    # -x1 - 2*x2 + 6 >= 0
    # -x1 + 2*x2 + 2 >= 0
    # x1 >= 0
    # x2 >= 0

    H = np.array([[2.0, 0.0], [0.0, 2.0]])
    C = np.array([[-2.0], [-5.0]])
    A = np.array([[1.0, -2.0], [-1.0, -2.0], [-1.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    B = np.array([[-2.0], [-6.0], [-2.0], [0.0], [0.0]])

    asm_test = ASM(H, C, A, B)
    asm_test.solve()
    print(asm_test.x)

    # draw graph
    fig, ax = plt.subplots()

    x0 = np.array([i[0] for i in asm_test.path])
    x1 = np.array([i[1] for i in asm_test.path])
    ax.plot(x0, x1, "go-")
    plt.show()


if __name__ == "__main__":
    main()