import torch
import torch.nn as nn
import torch.optim as optim

# Passo 1: Definir a classe da rede neural
class MinhaRedeNeural(nn.Module):
    def __init__(self):
        super(MinhaRedeNeural, self).__init__()
        # Definir as camadas da rede
        self.camada_oculta = nn.Linear(2, 3)  # Camada oculta com 2 entradas e 3 saídas
        self.activation = nn.ReLU()           # Função de ativação ReLU
        self.camada_saida = nn.Linear(3, 1)   # Camada de saída com 3 entradas e 1 saída
        
    def forward(self, x):
        # Definir o fluxo de dados na rede
        x = self.activation(self.camada_oculta(x))  # Aplicar camada oculta e ReLU
        x = self.camada_saida(x)                    # Aplicar camada de saída
        return x

# Passo 2: Criar uma instância do modelo
modelo = MinhaRedeNeural()

# Passo 3: Definir a função de perda e o otimizador
criterio = nn.MSELoss()                       # Função de perda: Mean Squared Error (MSE)
otimizador = optim.SGD(modelo.parameters(), lr=0.1)  # Otimizador: Stochastic Gradient Descent (SGD)

# Passo 4: Criar dados de exemplo (inputs e targets)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Passo 5: Loop de treinamento
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass (calcular previsões)
    outputs = modelo(X)
    loss = criterio(outputs, Y)  # Calcular a perda
    
    # Backward pass (calcular gradientes e atualizar pesos)
    otimizador.zero_grad()       # Zerar gradientes acumulados
    loss.backward()              # Calcular gradientes
    otimizador.step()            # Atualizar pesos
    
    # Imprimir métricas de desempenho
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Passo 6: Testar o modelo treinado
with torch.no_grad():
    saidas_teste = modelo(X)
    print('Saídas do modelo:')
    print(saidas_teste)
