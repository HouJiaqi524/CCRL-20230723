import torch

checkpoint = torch.load(r'out/MNIST/Noneopt_cor.pt')
print(checkpoint['epoch'])
print(checkpoint['train_loss'])
print(checkpoint['valid_cor'])