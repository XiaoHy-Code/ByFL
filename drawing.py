import numpy as np
import matplotlib.pyplot as plt

# 设定均值和标准差
mu = 0
sigma = 1

# 生成一些数据
x1 = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y1 = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x1-mu)**2/(2*sigma**2))



# 画图
plt.plot(x1, y1)
plt.plot(x1, y1*0.3)
plt.plot(x1, y1*0.7)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Normal distribution')
plt.show()