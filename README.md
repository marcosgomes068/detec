# Framework de Detecção de Intenções com Autoaprendizado

Um framework em Python para detecção de intenções em linguagem natural, com capacidades de autoaprendizado e integração com a API da Cohere. Otimizado para português brasileiro, mas facilmente adaptável para outros idiomas.

## Recursos

- 🧠 Detecção de intenções com alto grau de precisão
- 🔄 Autoaprendizado para melhorar continuamente
- 🌐 Suporte para português brasileiro
- 📊 API REST integrada para fácil uso em aplicações
- 🧩 Arquitetura modular e extensível
- 🔌 Integração com a API Cohere para melhor precisão
- 📈 Métricas e estatísticas de desempenho

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/marcospontoexe/intent-detection-framework-ptbr.git
   cd intent-detection-framework-ptbr
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure as variáveis de ambiente:
   ```bash
   cp .env.example .env
   ```

4. Edite o arquivo `.env` e adicione sua chave de API da Cohere:
   ```
   CO_API_KEY=sua_chave_da_cohere_aqui
   ```

## Uso

### Iniciando o Framework

```bash
python intent_detection_framework.py
```

### Usando como API

O framework inicia automaticamente um servidor API na porta 5000. Você pode testar com:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Olá, como vai você?"}'
```

### Usando como Biblioteca

```python
from intent_detection_framework import IntentDetectionFramework

# Instanciar framework
framework = IntentDetectionFramework(cohere_api_key="sua_chave_aqui")

# Treinar modelo
framework.train_model()

# Fazer predição
prediction = framework.predict_intent("Quero fazer um pedido")
print(f"Intenção: {prediction.intent}")
print(f"Confiança: {prediction.confidence}")
print(f"Entidades: {prediction.entities}")

# Iniciar autoaprendizado
framework.start_auto_learning(interval=600)  # 10 minutos
```

## Endpoints da API

- `GET /health` - Verifica se o serviço está operacional
- `GET /` - Documentação da API
- `POST /predict` - Detecta a intenção de um texto
- `POST /train` - Adiciona um exemplo de treinamento
- `POST /retrain` - Re-treina o modelo
- `GET /metrics` - Obtém métricas do modelo
- `POST /auto-learning/start` - Inicia o autoaprendizado
- `POST /auto-learning/stop` - Para o autoaprendizado

## Treinamento do Modelo

O framework vem pré-treinado com exemplos básicos, mas você pode adicionar seus próprios exemplos:

```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"text":"Me mostre o cardápio", "intent":"cardapio", "validated":true}'
```

## Exemplos de Cliente

### Cliente Terminal
O projeto inclui um cliente de terminal para testar o framework:

```bash
python chatbot_terminal.py
```

Este cliente exibe uma interface no terminal para conversar com o bot e testar as capacidades de detecção de intenções.

### Geração de Dados de Treinamento
Para gerar novos exemplos de treinamento:

```bash
python gerar_treinamento.py
```

### Tradução de Datasets
Para traduzir datasets de inglês para português:

```bash
python traduzir_dataset.py dataset.json -o dataset_ptbr.json
```

### Cliente Python

```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "Qual o preço do produto?"}
)
print(response.json())
```

## Intenções Suportadas

O framework vem pré-treinado com as seguintes intenções:

- `saudacao` - Saudações e cumprimentos
- `despedida` - Despedidas
- `agradecimento` - Agradecimentos
- `pedido` - Solicitações de pedidos
- `cardapio` - Perguntas sobre cardápio e opções 
- `cancelamento` - Cancelamento de pedidos
- `status_pedido` - Consultas sobre o status de um pedido
- `reclamacao` - Reclamações sobre produtos ou serviços
- `localizacao` - Perguntas sobre localização
- `horario` - Consultas sobre horário de funcionamento
- `pagamento` - Perguntas sobre formas de pagamento
- `editar_pedido` - Solicitações para alterar um pedido existente
- `troco` - Perguntas e solicitações relacionadas a troco
- `confirmacao` - Confirmação de informações

Você pode adicionar suas próprias intenções conforme necessário.

Inclui um cliente de terminal para teste interativo:

```bash
python chatbot_terminal.py
```

## Implementações Futuras

- Suporte para mais idiomas
- Melhorias no extrator de entidades
- Integração com outras APIs de LLM
- Interface web para gerenciamento
- Exportação e importação de modelos treinados

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autor

Desenvolvido por Marco
