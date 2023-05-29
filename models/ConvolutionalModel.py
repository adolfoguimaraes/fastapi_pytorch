import torch
from torch import nn

class ConvolutionalModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.convlayers = nn.Sequential(
        
        # Camada Convolucional com Kernel 3x3
        # Recebe de entrada 3 imagens (R, G e B) e gera 16 novas imagens
        nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),

        # Camada de ativação usando a função ReLU
        nn.ReLU(),

        # Max Pooling reduzindo as imagens de 32x32 para 16x16
        nn.MaxPool2d(2, 2),

        # Recebe de entrada 16 imagens e gera 32 como saída
        nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),

        # Camada de ativação usando a função ReLU
        nn.ReLU(),

        # Max Pooling reduzindo as imagens de 16x16 para 8x8 
        nn.MaxPool2d(2, 2),

      )

      self.linearlayers = nn.Sequential(
          nn.Linear(32*8*8, 256),
          nn.ReLU(),
          nn.Linear(256, 10)
      )

  def forward(self, x):
      x = self.convlayers(x)
      x = torch.flatten(x, 1)
      return self.linearlayers(x)