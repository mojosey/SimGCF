import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

def filter(x,alpha,beta,mu):
    y = mu/(1+torch.exp(alpha*(x+beta)))
    return y

# def filter(x,x1,w):
#     y = torch.ones(x.size())
#     high_index = torch.where(x < x1)[0]
#     low_index = torch.where(x >= x1)[0]
#
#     y[low_index] *= w[0]
#     y[high_index] *= w[1]
#     return y


def high_filter(x,k,i,o):
    y = o/(1+torch.exp(k*(x+i)))
    return y

def low_filter(x,k,i,o):
    y = o/(1+torch.exp(-k*(x+i)))
    return y

def fit_filter(x,a_list):
    y = 0
    for i in range(len(a_list)):
        y += a_list[i]*torch.pow(x,i)
    return y/len(a_list)

def JacobiConv(L, x, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    y = 0
    x1 = 1
    y += alphas[0]*x1
    coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
    coef2 = (a + b + 2) / (r - l)
    x2 = coef1  + coef2 *x
    y += alphas[1]*x2
    X = [x1,x2]
    for i in range(2,L):
        coef_l = 2 * i * (i + a + b) * (2 * i - 2 + a + b)
        coef_lm1_1 = (2 * i + a + b - 1) * (2 * i + a + b) * (2 * i + a + b - 2)
        coef_lm1_2 = (2 * i + a + b - 1) * (a**2 - b**2)
        coef_lm2 = 2 * (i - 1 + a) * (i - 1 + b) * (2 * i + a + b)
        tmp1 = (coef_lm1_1 / coef_l)
        tmp2 = (coef_lm1_2 / coef_l)
        tmp3 = (coef_lm2 / coef_l)
        tmp1_2 = tmp1 * (2 / (r - l))
        tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
        nx = tmp1_2 * (x*X[-1]) + tmp2_2 * X[-1]
        nx -= tmp3 * X[-2]
        X.append(nx)
        y += alphas[i]*nx
    y/= L
    return y

def Monomial(L,x,alphas):
    y = 0
    x1 = 1
    y += alphas[0] * x1
    X = [x1]
    for i in range(1, L):
        nx = x*X[-1]
        X.append(nx)
        y += alphas[i] * nx
    y /= L
    return y

def show_plot(X,save_path=None):
    fig, ax = plt.subplots()
    for x,y,label in X:
        ax.plot(x,y,label=label)

    # 添加标题和标签
    # ax.set_title('Function Plot with Axes')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # 在X轴上添加"Low", "Mid", "High"标签
    ax.text(-1, -1.22, 'High', ha='center', va='center', fontsize=20, color='black')
    ax.text(0, -1.22, 'Mid', ha='center', va='center', fontsize=20, color='black')
    ax.text(1, -1.22, 'Low', ha='center', va='center', fontsize=20, color='black')
    # 突出显示 x 和 y 坐标轴
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    # 设置 x 和 y 坐标轴的范围为 [-1, 1]
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.tick_params(axis='y', labelsize=15)  # 设置Y轴刻度的字体大小
    ax.tick_params(axis='x', labelsize=15)  # 设置X轴刻度的字体大小
    # 添加图例
    ax.legend(loc='lower right', prop={'size': 20})
    # 显示图像
    ax.grid(True)
    # plt.savefig(save_path)
    plt.show()

def fit_alphas(n,alpha,beta,mu,config):
    # alpha, beta, mu = config['alpha'], config['beta'], config['mu']
    a_list = torch.randn(n, requires_grad=True)
    optimizer = optim.Adam(params=[{"params": a_list}], lr=1e-3)

    train_x = torch.arange(-1, 1, 0.0001)
    # index = indices = torch.where((train_x >= -0.75) & (train_x <= 0.2))[0]
    # y = Monomial(n,train_x,alphas=[1 for _ in range(n)])
    y = JacobiConv(n, train_x, alphas=[1 for _ in range(n)], a=config['a'], b=config['b'])
    y_filter = filter(train_x, alpha, beta, mu)
    y_label = y_filter*y
    # y_label = y

    stop, step, best = 50, 0, 100000000
    for epoch in range(100000):
        y_pred = fit_filter(train_x, a_list)
        low_loss = torch.sqrt(torch.sum((y_pred - y_label) ** 2))

        loss = low_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("epoch: {:d} loss: {:.4f}".format(epoch, loss.item()))
        if loss < best:
            best = loss
            step = 0
            alphas = [float('{:.4f}'.format(i)) for i in a_list.detach().tolist()]
        else:
            step += 1

        if step == stop:
            break

    # show_plot([(train_x.numpy(), y_label.detach().numpy(), 'y_label'),
    #            (train_x.numpy(), y_pred.detach().numpy(), 'y_pred'),
    #            (train_x.numpy(), y.detach().numpy(), 'y')])
    print('alphas:', alphas)
    print("epoch: {:d} loss: {:.4f}".format(epoch, loss.item()))
    return alphas

if __name__=='__main__':
    config = {'alpha':-10, 'beta':0, 'mu':1,'a':1,'b':1.1}
    alphas = fit_alphas(4,alpha=config['alpha'],beta=config['beta'],mu=config['mu'],config=config)

