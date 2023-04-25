import numpy as np
from mnist import load_mnist


class MulLayer:
    def __init__(self):
        x = None
        y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy


class AddLayer:
    def __init__(self):
        x = None
        y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, dout):
        return dout, dout


# apple = 100
# apple_num = 2
# tax = 1.1

# n1 = MulLayer()
# n2 = MulLayer()

# n1.forward(n2.forward(100, 2), 1.1)

# a, b = n1.backward(1)
# print(n2.backward(a))


class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x
        out[self.mask] = 0
        return out

    def backward(self, dout):
        out = dout
        out[self.mask] = 0
        return out


# l = ReLu()
# l.forward(np.array([1, 2, 3, -0.5]))
# print(l.mask)
# print(l.backward(np.array([5, 6, 7, 8])))


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        out = dout * self.out * (1 - self.out)
        return out



class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

        self.db = None
        self.dw = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)

        # save gradients
        self.db = np.sum(dout, axis=0)
        self.dw = np.dot(self.x.T, dout)

        return dx

    def update(self, lr=0.01):
        if self.db is None or self.dw is None:
            raise ValueError()

        self.b -= self.db * lr
        self.W -= self.dw * lr


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(x, t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size

# class BatchNormalization:
#     def __init__(self):
#         self.x = None

#     def forward(self, x):
#         bathc_size = 


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.fc1 = Affine(
            np.random.randn(input_size, hidden_size) * weight_init_std,
            np.zeros(hidden_size),
        )
        self.relu = ReLu()
        self.fc2 = Affine(
            np.random.randn(hidden_size, output_size) * weight_init_std,
            np.zeros(output_size),
        )
        self.activation_with_loss = SoftmaxWithLoss()

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        return x

    # def loss(self, x, t):
    #     y = self.predict(x)
    #     return self.activation_with_loss(y, t)

    def accuracy(self, x, t):
        preds = self.forward(x)
        preds = np.argmax(preds, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(preds == t) / float(x.shape[0])

    def backward(self, x, t):
        y = self.forward(x)
        self.activation_with_loss.forward(y, t)

        # backward
        dout = 1
        dout = self.activation_with_loss.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.fc1.backward(dout)

    def update(self, lr):
        self.fc1.update(lr=lr)
        self.fc2.update(lr=lr)


def train():
    epochs = 15
    batch_size = 100
    lr = 0.1

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )
    model = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    loss_log = []
    accuracy_log = []

    for _ in range(epochs):
        # Create batches

        # batch_mask = np.random.choice(x_train.shape[0], batch_size)
        batch_mask = np.arange(x_train.shape[0])
        np.random.shuffle(batch_mask)

        dataloader = []
        for i in range((x_train.shape[0] // batch_size) -1 ):
            batched_data = x_train[batch_mask[i * batch_size : (i + 1) * batch_size]]
            batched_labels = t_train[batch_mask[i * batch_size : (i + 1) * batch_size]]
            dataloader.append((batched_data, batched_labels))

        for data, label in dataloader:
            model.backward(data, label)
            model.update(lr=lr)

        pred = model.forward(data)
        loss = cross_entropy_error(softmax(pred), label)
        loss_log.append(loss)
        accuracy = model.accuracy(data, label)
        accuracy_log.append(accuracy)

        print(f"{accuracy=} {loss=}")


train()
