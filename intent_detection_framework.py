# Framework de Detecção de Intenções com Autoaprendizado
# Tecnologias: Python, Cohere, NLTK, scikit-learn, SQLite

import sqlite3
import json
import numpy as np
import pickle
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from flask import Flask, request, jsonify
from flask_cors import CORS
import cohere
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import re
import threading
import time

# Baixar dados necessários do NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

@dataclass
class IntentPrediction:
    """Classe para representar uma predição de intenção"""
    intent: str
    confidence: float
    entities: Dict[str, str]
    raw_text: str

class DatabaseManager:
    """Gerenciador do banco de dados SQLite"""
    
    def __init__(self, db_path: str = "intent_framework.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa as tabelas do banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de intenções de treinamento
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                intent TEXT NOT NULL,
                entities TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validated BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Tabela de logs de predições
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                predicted_intent TEXT,
                confidence REAL,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de métricas do modelo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                accuracy REAL,
                model_version TEXT,
                training_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                cohere_suggestion TEXT,
                cohere_explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validated BOOLEAN DEFAULT FALSE,
                validated_intent TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_training_data(self, text: str, intent: str, entities: Dict = None):
        """Insere dados de treinamento"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        entities_json = json.dumps(entities) if entities else None
        cursor.execute('''
            INSERT INTO training_data (text, intent, entities)
            VALUES (?, ?, ?)
        ''', (text, intent, entities_json))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self) -> List[Tuple[str, str]]:
        """Recupera dados de treinamento"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT text, intent FROM training_data WHERE validated = TRUE')
        data = cursor.fetchall()
        
        conn.close()
        return data
    
    def log_prediction(self, text: str, intent: str, confidence: float, feedback: str = None):
        """Registra uma predição"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prediction_logs (text, predicted_intent, confidence, feedback)
            VALUES (?, ?, ?, ?)
        ''', (text, intent, confidence, feedback))
        
        conn.commit()
        conn.close()

    def insert_pending_example(self, text: str, cohere_suggestion: str, cohere_explanation: str):
        """Insere exemplo pendente de validação"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                cohere_suggestion TEXT,
                cohere_explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validated BOOLEAN DEFAULT FALSE,
                validated_intent TEXT
            )
        ''')
        cursor.execute('''
            INSERT INTO pending_examples (text, cohere_suggestion, cohere_explanation)
            VALUES (?, ?, ?)
        ''', (text, cohere_suggestion, cohere_explanation))
        conn.commit()
        conn.close()

    def list_pending_examples(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, text, cohere_suggestion, cohere_explanation, validated, validated_intent FROM pending_examples WHERE validated = FALSE')
        data = cursor.fetchall()
        conn.close()
        return data

    def validate_example(self, example_id: int, intent: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE pending_examples SET validated = TRUE, validated_intent = ? WHERE id = ?
        ''', (intent, example_id))
        # Também adiciona ao dataset de treinamento
        cursor.execute('SELECT text FROM pending_examples WHERE id = ?', (example_id,))
        row = cursor.fetchone()
        if row:
            self.insert_training_data(row[0], intent)
        conn.commit()
        conn.close()

class TextPreprocessor:
    """Pré-processador de texto"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('portuguese') + stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def preprocess(self, text: str) -> str:
        """Pré-processa um texto"""
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais
        text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
        
        # Tokenizar
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remover stop words e aplicar stemming
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)

class EntityExtractor:
    """Extrator de entidades simples"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{2,3}[-.\s]?\d{4,5}[-.\s]?\d{4}\b',
            'cpf': r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
            'money': r'R\$\s?\d+(?:[.,]\d{2})?',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }
    
    def extract(self, text: str) -> Dict[str, str]:
        """Extrai entidades do texto"""
        entities = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches
        
        return entities

class CohereIntentClassifier:
    """Classificador de intenções usando Cohere"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if api_key:
            self.co = cohere.Client(api_key)
        else:
            self.co = None
    
    def classify_with_cohere(self, text: str, examples: List[Dict]) -> Dict:
        """Classifica usando embeddings do Cohere em vez do Classify API que exige fine-tuning"""
        if not self.co:
            return {"intent": "unknown", "confidence": 0.0}
        
        try:
            # Extrair textos e labels dos exemplos
            texts = [text] + [example["text"] for example in examples]
            
            # Gerar embeddings para o texto e exemplos
            response = self.co.embed(
                texts=texts,
                model="embed-multilingual-v3.0",
                input_type="search_query" if texts.index(text) == 0 else "search_document"
            )
            
            if not response.embeddings:
                return {"intent": "unknown", "confidence": 0.0}
            
            # Obter embedding do texto de entrada
            query_embedding = response.embeddings[0]
            
            # Calcular similaridade com cada exemplo
            similarities = []
            for i, example in enumerate(examples):
                # O embedding do exemplo está no índice i+1 (já que o texto de query está no índice 0)
                example_embedding = response.embeddings[i+1]
                
                # Calcular similaridade de cosseno
                similarity = self._cosine_similarity(query_embedding, example_embedding)
                similarities.append({
                    "intent": example["label"],
                    "similarity": similarity
                })
            
            # Encontrar a intenção com maior similaridade
            if similarities:
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
                best_match = similarities[0]
                
                return {
                    "intent": best_match["intent"],
                    "confidence": best_match["similarity"]
                }
            
        except Exception as e:
            print(f"Erro na classificação Cohere: {e}")
        
        return {"intent": "unknown", "confidence": 0.0}
    
    def _cosine_similarity(self, vec1, vec2):
        """Calcula a similaridade de cosseno entre dois vetores"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class IntentDetectionFramework:
    """Framework principal de detecção de intenções"""
    
    def __init__(self, cohere_api_key: str = None, db_path: str = "intent_framework.db"):
        self.db = DatabaseManager(db_path)
        self.preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor()
        self.cohere_classifier = CohereIntentClassifier(cohere_api_key)
        
        # Modelo scikit-learn
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.model_trained = False
        
        # Thread para autoaprendizado
        self.auto_learning_active = False
        self.learning_thread = None
        
        # Dados iniciais de exemplo
        self._init_sample_data()
    
    def _init_sample_data(self):
        """Inicializa dados de exemplo"""
        sample_data = [
            ("Olá, como você está?", "saudacao"),
            ("Oi, tudo bem?", "saudacao"),
            ("Bom dia!", "saudacao"),
            ("Como posso fazer um pedido?", "pedido"),
            ("Quero comprar um produto", "pedido"),
            ("Gostaria de fazer uma compra", "pedido"),
            ("Qual o preço deste item?", "preco"),
            ("Quanto custa?", "preco"),
            ("Valor do produto", "preco"),
            ("Preciso cancelar meu pedido", "cancelamento"),
            ("Como cancelo?", "cancelamento"),
            ("Quero cancelar", "cancelamento"),
            ("Obrigado pela ajuda", "agradecimento"),
            ("Muito obrigado!", "agradecimento"),
            ("Valeu!", "agradecimento"),
            ("Tchau", "despedida"),
            ("Até logo", "despedida"),
            ("Adeus", "despedida")
        ]
        
        # Inserir dados de exemplo se não existirem
        existing_data = self.db.get_training_data()
        if len(existing_data) == 0:
            for text, intent in sample_data:
                self.db.insert_training_data(text, intent)
                # Marcar como validado
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('UPDATE training_data SET validated = TRUE WHERE text = ?', (text,))
                conn.commit()
                conn.close()
    
    def train_model(self):
        """Treina o modelo de classificação"""
        training_data = self.db.get_training_data()
        
        if len(training_data) < 2:
            print("Dados insuficientes para treinamento")
            return False
        
        texts, labels = zip(*training_data)
        
        # Pré-processar textos
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Vetorizar
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Treinar classificador
        self.classifier.fit(X, labels)
        self.model_trained = True
        
        # Calcular métricas se possível
        if len(set(labels)) > 1 and len(texts) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Salvar métricas
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_metrics (accuracy, model_version, training_size)
                VALUES (?, ?, ?)
            ''', (accuracy, "v1.0", len(training_data)))
            conn.commit()
            conn.close()
            
            print(f"Modelo treinado com acurácia: {accuracy:.2f}")
        
        return True
    
    def predict_intent(self, text: str) -> IntentPrediction:
        """Prediz a intenção de um texto"""
        # Extrair entidades
        entities = self.entity_extractor.extract(text)
        
        # Predição com modelo local
        local_intent = "unknown"
        local_confidence = 0.0
        
        if self.model_trained:
            processed_text = self.preprocessor.preprocess(text)
            X = self.vectorizer.transform([processed_text])
            
            # Predição
            predicted_intent = self.classifier.predict(X)[0]
            predicted_proba = self.classifier.predict_proba(X)[0]
            
            local_intent = predicted_intent
            local_confidence = max(predicted_proba)
        
        # Predição com Cohere (se disponível)
        cohere_result = {"intent": "unknown", "confidence": 0.0}
        if self.cohere_classifier.co:
            training_data = self.db.get_training_data()
            if training_data:
                examples = [{"text": t, "label": i} for t, i in training_data[:50]]  # Limitar exemplos
                cohere_result = self.cohere_classifier.classify_with_cohere(text, examples)
        
        # Combinar resultados (priorizar o de maior confiança)
        if cohere_result["confidence"] > local_confidence:
            final_intent = cohere_result["intent"]
            final_confidence = cohere_result["confidence"]
        else:
            final_intent = local_intent
            final_confidence = local_confidence
        
        # Criar predição
        prediction = IntentPrediction(
            intent=final_intent,
            confidence=final_confidence,
            entities=entities,
            raw_text=text
        )
        
        # Registrar predição
        self.db.log_prediction(text, final_intent, final_confidence)
        
        return prediction
    
    def add_training_example(self, text: str, intent: str, entities: Dict = None, validated: bool = False):
        """Adiciona um exemplo de treinamento"""
        self.db.insert_training_data(text, intent, entities)
        
        if validated:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE training_data SET validated = TRUE WHERE text = ? AND intent = ?', (text, intent))
            conn.commit()
            conn.close()
    
    def start_auto_learning(self, interval: int = 300):  # 5 minutos
        """Inicia o processo de autoaprendizado"""
        self.auto_learning_active = True
        self.learning_thread = threading.Thread(target=self._auto_learning_loop, args=(interval,))
        self.learning_thread.daemon = True
        self.learning_thread.start()
        print("Autoaprendizado iniciado")
    
    def stop_auto_learning(self):
        """Para o processo de autoaprendizado"""
        self.auto_learning_active = False
        if self.learning_thread:
            self.learning_thread.join()
        print("Autoaprendizado parado")
    
    def _auto_learning_loop(self, interval: int):
        """Loop de autoaprendizado"""
        while self.auto_learning_active:
            try:
                # Verificar se há novos dados validados
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM training_data WHERE validated = TRUE')
                validated_count = cursor.fetchone()[0]
                conn.close()
                
                # Re-treinar se houver dados suficientes
                if validated_count >= 5:  # Mínimo de dados
                    print("Re-treinando modelo com novos dados...")
                    self.train_model()
                
                time.sleep(interval)
            except Exception as e:
                print(f"Erro no autoaprendizado: {e}")
                time.sleep(interval)
    
    def get_metrics(self) -> Dict:
        """Obtém métricas do modelo"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Métricas do modelo
        cursor.execute('SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1')
        model_metrics = cursor.fetchone()
        
        # Estatísticas gerais
        cursor.execute('SELECT COUNT(*) FROM training_data')
        total_training = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM prediction_logs')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT predicted_intent) FROM prediction_logs')
        unique_intents = cursor.fetchone()[0]
        
        conn.close()
        
        metrics = {
            "total_training_data": total_training,
            "total_predictions": total_predictions,
            "unique_intents": unique_intents,
            "model_trained": self.model_trained,
            "auto_learning_active": self.auto_learning_active
        }
        
        if model_metrics:
            metrics.update({
                "last_accuracy": model_metrics[1],
                "model_version": model_metrics[2],
                "last_training_size": model_metrics[3]
            })
        
        return metrics
    
    def cohere_suggest_intent(self, text: str) -> dict:
        """Usa a Cohere para sugerir intenção e explicação para uma frase desconhecida"""
        if not self.co:
            return {"suggestion": "unknown", "explanation": "Cohere não configurado."}
        prompt = f"""
Você é um classificador de intenções para um chatbot de atendimento em português. Dada a frase do usuário, sugira a intenção (apenas o nome da intenção, como 'pedido', 'cancelamento', 'pagamento', etc) e explique o motivo da sugestão.

Frase: \"{text}\"
Resposta (formato JSON): {{\"intent\": <intenção>, \"explicacao\": <explicação curta>}}
"""
        try:
            response = self.co.generate(
                model="command-nightly",
                prompt=prompt,
                max_tokens=100,
                temperature=0.2
            )
            import json as _json
            result = response.generations[0].text.strip()
            # Extrai JSON da resposta
            if result.startswith('{'):
                data = _json.loads(result)
                return {"suggestion": data.get("intent", "unknown"), "explanation": data.get("explicacao", "")}
            else:
                return {"suggestion": "unknown", "explanation": result}
        except Exception as e:
            return {"suggestion": "unknown", "explanation": str(e)}

# API REST Flask
app = Flask(__name__)
CORS(app)

# Instância global do framework
framework = IntentDetectionFramework()

@app.route('/', methods=['GET'])
def api_root():
    """Endpoint raiz com documentação da API"""
    api_docs = {
        "name": "Framework de Detecção de Intenções com Autoaprendizado",
        "version": "1.0",
        "description": "API para detecção de intenções com autoaprendizado",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Documentação da API"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check da API"
            },
            {
                "path": "/predict",
                "method": "POST",
                "description": "Predição de intenção a partir de um texto",
                "body": {"text": "string"}
            },
            {
                "path": "/train",
                "method": "POST",
                "description": "Adicionar dados de treinamento",
                "body": {
                    "text": "string", 
                    "intent": "string", 
                    "entities": "object (opcional)", 
                    "validated": "boolean (opcional)"
                }
            },
            {
                "path": "/retrain",
                "method": "POST",
                "description": "Re-treinar modelo"
            },
            {
                "path": "/metrics",
                "method": "GET",
                "description": "Obter métricas do modelo e sistema"
            },
            {
                "path": "/auto-learning/start",
                "method": "POST",
                "description": "Iniciar autoaprendizado",
                "body": {"interval": "number (opcional)"}
            },
            {
                "path": "/auto-learning/stop",
                "method": "POST",
                "description": "Parar autoaprendizado"
            },
            {
                "path": "/pending-examples",
                "method": "GET",
                "description": "Listar exemplos pendentes de validação"
            },
            {
                "path": "/validate-example",
                "method": "POST",
                "description": "Validar exemplo pendente",
                "body": {
                    "id": "integer",
                    "intent": "string"
                }
            }
        ]
    }
    
    return jsonify(api_docs)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check da API"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict_intent_api():
    """Endpoint para predição de intenção"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Texto não fornecido"}), 400
        
        prediction = framework.predict_intent(text)
        
        return jsonify({
            "intent": prediction.intent,
            "confidence": prediction.confidence,
            "entities": prediction.entities,
            "text": prediction.raw_text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def add_training_data_api():
    """Endpoint para adicionar dados de treinamento"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        intent = data.get('intent', '')
        entities = data.get('entities', {})
        validated = data.get('validated', False)
        
        if not text or not intent:
            return jsonify({"error": "Texto e intenção são obrigatórios"}), 400
        
        framework.add_training_example(text, intent, entities, validated)
        
        return jsonify({"message": "Dados de treinamento adicionados com sucesso"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model_api():
    """Endpoint para re-treinar o modelo"""
    try:
        success = framework.train_model()
        
        if success:
            return jsonify({"message": "Modelo re-treinado com sucesso"})
        else:
            return jsonify({"error": "Falha no treinamento - dados insuficientes"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics_api():
    """Endpoint para obter métricas"""
    try:
        metrics = framework.get_metrics()
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auto-learning/start', methods=['POST'])
def start_auto_learning_api():
    """Endpoint para iniciar autoaprendizado"""
    try:
        data = request.get_json() or {}
        interval = data.get('interval', 300)  # 5 minutos padrão
        
        framework.start_auto_learning(interval)
        return jsonify({"message": "Autoaprendizado iniciado"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auto-learning/stop', methods=['POST'])
def stop_auto_learning_api():
    """Endpoint para parar autoaprendizado"""
    try:
        framework.stop_auto_learning()
        return jsonify({"message": "Autoaprendizado parado"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pending-examples', methods=['GET'])
def list_pending_examples():
    """Lista exemplos pendentes de validação"""
    data = framework.db.list_pending_examples()
    return jsonify([
        {
            "id": row[0],
            "text": row[1],
            "cohere_suggestion": row[2],
            "cohere_explanation": row[3],
            "validated": bool(row[4]),
            "validated_intent": row[5]
        } for row in data
    ])

@app.route('/validate-example', methods=['POST'])
def validate_example():
    """Valida um exemplo pendente e adiciona ao dataset"""
    req = request.get_json()
    example_id = req.get('id')
    intent = req.get('intent')
    if not example_id or not intent:
        return jsonify({"error": "id e intent são obrigatórios"}), 400
    framework.db.validate_example(example_id, intent)
    return jsonify({"status": "validado"})

def main():
    """Função principal"""
    print("🧠 Framework de Detecção de Intenções com Autoaprendizado")
    print("=" * 60)
    
    # Carregar variáveis de ambiente do arquivo .env
    load_dotenv()
    cohere_api_key = os.getenv("CO_API_KEY")
    
    if cohere_api_key:
        print("✅ Chave de API da Cohere encontrada no arquivo .env")
        # Atualizar a instância global do framework com a chave da Cohere
        global framework
        framework = IntentDetectionFramework(cohere_api_key=cohere_api_key)
    else:
        print("⚠️ Chave de API da Cohere não encontrada no arquivo .env")
        
    # Treinar modelo inicial
    print("Treinando modelo inicial...")
    framework.train_model()
    
    # Exemplos de uso
    print("\n📝 Testando predições:")
    test_texts = [
        "Oi, tudo bem?",
        "Quero fazer um pedido",
        "Quanto custa este produto?",
        "Preciso cancelar",
        "Obrigado pela ajuda"
    ]
    
    for text in test_texts:
        prediction = framework.predict_intent(text)
        print(f"Texto: '{text}'")
        print(f"Intenção: {prediction.intent} (confiança: {prediction.confidence:.2f})")
        if prediction.entities:
            print(f"Entidades: {prediction.entities}")
        print("-" * 40)
    
    # Mostrar métricas
    print("\n📊 Métricas do sistema:")
    metrics = framework.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Iniciar autoaprendizado
    print("\n🔄 Iniciando autoaprendizado...")
    # framework.start_auto_learning()
    
    # Iniciar API
    print("\n🚀 Iniciando API REST...")
    print("Endpoints disponíveis:")
    print("- POST /predict - Predição de intenção")
    print("- POST /train - Adicionar dados de treinamento")
    print("- POST /retrain - Re-treinar modelo")
    print("- GET /metrics - Obter métricas")
    print("- POST /auto-learning/start - Iniciar autoaprendizado")
    print("- POST /auto-learning/stop - Parar autoaprendizado")
    print("- GET /pending-examples - Listar exemplos pendentes")
    print("- POST /validate-example - Validar exemplo pendente")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()

# Exemplo de uso como biblioteca:
"""
# Instanciar framework
framework = IntentDetectionFramework(cohere_api_key="sua_chave_aqui")

# Treinar modelo
framework.train_model()

# Fazer predição
prediction = framework.predict_intent("Quero fazer um pedido")
print(f"Intenção: {prediction.intent}")
print(f"Confiança: {prediction.confidence}")
print(f"Entidades: {prediction.entities}")

# Adicionar dados de treinamento
framework.add_training_example("Gostaria de comprar", "pedido", validated=True)

# Re-treinar
framework.train_model()

# Iniciar autoaprendizado
framework.start_auto_learning(interval=600)  # 10 minutos
"""