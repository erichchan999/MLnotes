import torch
import torch.nn as nn
from torch import optim

result = []

A = torch.tensor([[1,2,1,-1],[-1,1,0,2],[0,-1,-2,1]]).float()
b = torch.tensor([3,2,-2]).float()
gamma = 0.2
alpha = 0.1

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.ones(A.shape[1], requires_grad=True))

    def forward(self, M):
        return M @ self.x

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=alpha)

result.append(model.x.data.detach().clone())

terminationCond = False
while not terminationCond:
    bhat = model.forward(A)
    regularisation_term = (gamma / 2) * torch.linalg.norm(model.x, ord=2) ** 2
    loss = (1 / 2) * torch.linalg.norm(bhat - b) ** 2 + regularisation_term
    loss.backward()
    optimizer.step()

    result.append(model.x.data.detach().clone())

    if torch.linalg.norm(model.x.grad) < 0.001:
        terminationCond = True

    optimizer.zero_grad()

print(result[:5])
print(result[-5:])