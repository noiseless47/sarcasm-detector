import cv2
import numpy as np
import os
import json
import torch
import re
import sys
import traceback
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoConfig,
    LlamaForSequenceClassification, 
    LlamaTokenizer
)
from torch import nn
import torch.nn.functional as F
from llama_cpp import Llama
import sentencepiece as spm
from .models.sarcasm_neural_network import SarcasmNeuralNetwork
from .models.llama_model import LlamaModel
from .model_manager import ModelManager
from .cloud_config import LLAMA_MODEL_PATH

# Add the LLaMA model directory to Python path
LLAMA_MODEL_PATH = Path(__file__).parent / "models" / "Llama3.2-3B-Instruct"
sys.path.append(str(LLAMA_MODEL_PATH))

def process_image(image):
    # Placeholder for image processing
    return "50% Clown 🤡", "Image analysis coming soon! 🎪"

class SarcasmAnalyzer:
    """Analyzer for sarcasm detection."""
    def __init__(self):
        print("Initializing SarcasmAnalyzer...")
        
        # Load parameters from params.json
        params_path = LLAMA_MODEL_PATH / "params.json"
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Reduce model size to fit in 6GB VRAM
        params['dim'] = 768  # Reduced from 3072
        params['n_layers'] = 12  # Reduced from 28
        params['n_heads'] = 12  # Reduced from 24
        print(f"Using reduced parameters for memory efficiency: {params}")
        
        # Initialize model with reduced size
        self.model = SarcasmNeuralNetwork(params)
        
        # Initialize tokenizer
        tokenizer_path = str(LLAMA_MODEL_PATH)
        try:
            print(f"Loading LlamaTokenizer from {tokenizer_path}")
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
            print("Successfully loaded LlamaTokenizer")
        except Exception as e:
            print(f"Error loading LlamaTokenizer: {e}")
            print("Falling back to BERT tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Use mixed precision and memory efficient settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # Enable automatic mixed precision
            self.scaler = torch.cuda.amp.GradScaler()
            # Empty CUDA cache
            torch.cuda.empty_cache()
            print("CUDA enabled with mixed precision")
        
        print(f"Using device: {self.device}")
        
        # Move model to device with memory efficient settings
        self.model = self.model.to(self.device)
        if hasattr(self.model, 'half'):
            self.model = self.model.half()  # Convert to FP16
        
        self.confidence_threshold = 0.6
        self.load_sarcasm_dataset()
        self.load_or_train_model()
        print("SarcasmAnalyzer initialized successfully!")

    def analyze_text(self, text):
        """Analyze text for sarcasm with significantly improved accuracy."""
        # Store the text for contextual processing
        self.last_analyzed_text = text
        print(f"Analyzing text: {text}")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract features for explanation
        features = self.extract_contextual_features(text)
        
        # Calculate feature-based sarcasm score (rule-based approach)
        feature_score = self.calculate_feature_score(features)
        
        # Pre-check for obvious sarcasm patterns
        obvious_sarcasm = self.check_obvious_sarcasm_patterns(text)
        if obvious_sarcasm:
            combined_score = 0.85  # High confidence for obvious sarcasm
        else:
            # Tokenize and get model prediction
            try:
                inputs = self.tokenizer(
                    processed_text, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=128
                )
                
                # Only pass input_ids and attention_mask to the model
                model_inputs = {
                    'input_ids': inputs['input_ids'].to(self.device)
                }
                if 'attention_mask' in inputs:
                    model_inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
                
                # Get model prediction
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**model_inputs)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    probabilities = torch.softmax(logits, dim=1)
                    model_prediction = probabilities[0][1].item()  # Probability of being sarcastic
            except Exception as e:
                print(f"Error during model prediction: {e}")
                traceback.print_exc()
                # Fall back to feature-based score only
                model_prediction = 0.5  # Neutral
            
            # Combine model prediction with feature score (weighted average)
            # Give more weight to feature score since our model is limited
            combined_score = 0.2 * model_prediction + 0.8 * feature_score
            
            # Apply final adjustments based on special patterns
            combined_score = self.apply_final_adjustments(text, combined_score, features)
        
        # Format rating and generate explanation
        if combined_score > 0.5:  # Sarcastic
            # Scale confidence to be more decisive
            confidence = 0.5 + (combined_score - 0.5) * 1.5  # Amplify difference from 0.5
            confidence = min(confidence, 0.95)  # Cap at 95%
            
            rating = f"{int(confidence * 100)}% Sarcastic 🙄"
            explanation = self._generate_sarcasm_explanation(confidence, features, text)
        else:  # Sincere
            # Scale confidence to be more decisive
            confidence = 0.5 + (0.5 - combined_score) * 1.5  # Amplify difference from 0.5
            confidence = min(confidence, 0.95)  # Cap at 95%
            
            rating = f"{int(confidence * 100)}% Sincere 😊"
            explanation = self._generate_sincere_explanation(confidence, text)
        
        print(f"Response: {{'rating': '{rating}', 'explanation': '{explanation}'}}")
        return rating, explanation

    def check_obvious_sarcasm_patterns(self, text):
        """Check for patterns that are almost always sarcastic."""
        text_lower = text.lower()
        
        # Classic sarcastic openings followed by positive words
        sarcastic_openings = [
            "oh great", "oh wonderful", "oh perfect", "oh fantastic", 
            "well that's just great", "just what i needed", "perfect timing",
            "how convenient", "lucky me", "my favorite part"
        ]
        
        # Common highly sarcastic phrases
        obvious_phrases = [
            "yeah right", "as if", "in your dreams", "tell me about it",
            "story of my life", "big surprise", "color me surprised", 
            "who would have thought", "shocking revelation",
            "i'm shocked, shocked i tell you", "surprise surprise",
            "because that makes sense", "obviously the best choice",
            "nothing could possibly go wrong", "what could go wrong"
        ]
        
        # Exaggeration patterns
        exaggeration_patterns = [
            r"(sooo+|soooo+) (fun|great|awesome|perfect|wonderful)",
            r"(absolutely|totally|completely) (thrilled|delighted|overjoyed) by",
            r"(best|worst) (day|thing) ever",
            r"couldn't be (happier|more excited|more thrilled)"
        ]
        
        # Check for obvious phrases
        if any(phrase in text_lower for phrase in obvious_phrases):
            return True
        
        # Check for sarcastic openings
        if any(text_lower.startswith(opening) for opening in sarcastic_openings):
            return True
        
        # Check for exaggeration patterns
        for pattern in exaggeration_patterns:
            if re.search(pattern, text_lower):
                return True
            
        # Positive sentiment with multiple exclamation/question marks is often sarcastic
        if re.search(r"(great|wonderful|perfect|fantastic|excellent|amazing).*[!?]{2,}", text_lower):
            return True
        
        return False

    def apply_final_adjustments(self, text, score, features):
        """Apply final adjustments to the score based on additional patterns."""
        text_lower = text.lower()
        
        # List of common literal statements
        literal_statements = [
            "have a good day", "hope you're well", "thank you", "thanks for",
            "appreciate your", "looking forward to", "excited to", "happy to",
            "glad to", "hope this helps", "let me know if", "please let me know",
            "congratulations on", "good luck with", "best wishes", "all the best",
            "sincerely", "good morning", "good afternoon", "good evening"
        ]
        
        # Educational or explanatory context
        educational_markers = [
            "according to", "research shows", "studies have found", "evidence suggests",
            "experts say", "data indicates", "fact is", "truth is", "reality is",
            "historically", "scientifically", "statistically", "in summary", "to summarize"
        ]
        
        # Common factual statements
        factual_markers = [
            "is located in", "was founded in", "consists of", "is made of",
            "was created by", "is composed of", "contains", "includes", "requires"
        ]
        
        # Strong sincere context
        if any(marker in text_lower for marker in literal_statements):
            score = max(score - 0.3, 0.2)  # Strongly reduce sarcasm score
        
        # Educational context is usually sincere
        if any(marker in text_lower for marker in educational_markers):
            score = max(score - 0.25, 0.25)
        
        # Factual statements are usually sincere
        if any(marker in text_lower for marker in factual_markers):
            score = max(score - 0.2, 0.3)
        
        # Very short statements (less than 4 words) are less likely to be sarcastic
        if len(text.split()) < 4 and score < 0.7:
            score = max(score - 0.15, 0.3)
        
        # First-person statements with positive sentiment are less likely to be sarcastic
        first_person_positive = re.search(r"i (am|feel|love|like|enjoy|appreciate)", text_lower)
        if first_person_positive and not features.get('has_punctuation_emphasis'):
            score = max(score - 0.2, 0.3)
        
        # Direct questions are less likely to be sarcastic (unless rhetorical)
        if text_lower.startswith(("what", "when", "where", "who", "how")) and text.endswith("?") and not features.get('has_rhetorical_question'):
            score = max(score - 0.2, 0.3)
        
        return score

    def extract_contextual_features(self, text):
        """Enhanced contextual feature extraction with more sensitive sarcasm detection."""
        # Context patterns (expanded)
        context_patterns = {
            'contrast_markers': {
                'but', 'however', 'although', 'though', 'yet', 'nevertheless',
                'surprisingly', 'unexpectedly', 'ironically', 'funny how',
                'imagine that', 'shock', 'shocking', 'supposedly', 'apparently',
                'just what i needed', 'exactly what i needed', 'perfect timing',
                'suddenly', 'miraculously', 'somehow', 'magically', 'conveniently'
            },
            'emphasis_markers': {
                'very', 'so', 'such', 'quite', 'rather', 'absolutely', 'totally',
                'completely', 'utterly', 'literally', 'actually', 'seriously',
                'just', 'really', 'exactly', 'precisely', 'certainly', 'definitely',
                'beyond', 'incredibly', 'amazingly', 'extraordinarily'
            },
            'mock_praise': {
                'genius', 'brilliant', 'outstanding', 'impressive', 'remarkable',
                'extraordinary', 'exceptional', 'magnificent', 'splendid', 'great',
                'wonderful', 'amazing', 'fantastic', 'excellent', 'perfect', 'ideal',
                'stellar', 'phenomenal', 'superb', 'spectacular', 'tremendous', 'bravo'
            },
            'understatement': {
                'slightly', 'somewhat', 'maybe', 'perhaps', 'possibly',
                'presumably', 'supposedly', 'allegedly', 'kind of', 'sort of',
                'a bit', 'a little', 'marginally', 'barely', 'hardly', 'scarcely'
            },
            'exaggeration': {
                'always', 'never', 'everyone', 'nobody', 'everything', 'nothing',
                'every time', 'all the time', 'constantly', 'eternally', 'forever',
                'entire', 'absolute', 'ultimate', 'extreme', 'countless', 'infinite'
            }
        }
        
        # Expanded sentiment analysis
        sentiment_patterns = {
            'positive': {
                'great', 'wonderful', 'amazing', 'fantastic', 'brilliant',
                'love', 'awesome', 'excellent', 'perfect', 'delightful',
                'outstanding', 'incredible', 'superb', 'magnificent', 'happy',
                'glad', 'pleased', 'thrilled', 'excited', 'joyful', 'content',
                'satisfied', 'gratified', 'thankful', 'grateful', 'blessed'
            },
            'negative': {
                'worst', 'terrible', 'awful', 'horrible', 'disaster',
                'hate', 'boring', 'pathetic', 'ridiculous', 'useless',
                'disappointing', 'mediocre', 'failure', 'mess', 'annoying',
                'frustrating', 'inconvenient', 'unstable', 'painful', 'awful',
                'dreadful', 'miserable', 'unbearable', 'insufferable', 'vexing',
                'infuriating', 'maddening', 'disturbing', 'appalling', 'atrocious'
            }
        }

        text_lower = text.lower()
        words = set(text_lower.split())
        sentences = text.split('.')
        
        # Check for excessive punctuation (more than 1 of the same type in a row)
        excessive_punctuation = bool(re.search(r'[!?]{2,}', text))
        
        # Check for ALLCAPS words (not acronyms)
        all_caps_words = [word for word in text.split() if word.isupper() and len(word) > 2]
        has_all_caps = len(all_caps_words) > 0
        
        # Enhanced feature extraction
        features = {
            'has_contrast': any(marker in text_lower for marker in context_patterns['contrast_markers']),
            'has_emphasis': any(marker in text_lower for marker in context_patterns['emphasis_markers']),
            'has_mock_praise': any(marker in text_lower for marker in context_patterns['mock_praise']),
            'has_understatement': any(marker in text_lower for marker in context_patterns['understatement']),
            'has_exaggeration': any(marker in text_lower for marker in context_patterns['exaggeration']),
            'has_sentiment_contrast': bool(words & sentiment_patterns['positive']) and bool(words & sentiment_patterns['negative']),
            'has_punctuation_emphasis': excessive_punctuation or '!?' in text or '...' in text,
            'has_quotes': '"' in text or "'" in text,
            'has_capitalization': has_all_caps,
            'sentence_length': len(sentences),
            'has_negation': any(neg in text_lower for neg in ['not', 'never', 'no', 'none', "n't", "cannot"]),
            'has_rhetorical_question': '?' in text and any(q.strip().startswith(('why', 'how', 'what', 'really', 'seriously', 'exactly', 'like')) for q in text.split('?')),
            'repeated_words': any(text_lower.split().count(word.lower()) > 1 for word in words),
            'has_negative_context': bool(words & sentiment_patterns['negative']) or any(
                neg in text_lower for neg in ['problem', 'issue', 'trouble', 'difficult', 'hard', 'challenge']
            ),
        }
        
        # Check for common sarcastic phrases with more comprehensive patterns
        sarcastic_phrases = [
            "yeah right", "oh really", "sure", "totally", "as if", 
            "what a surprise", "no way", "you don't say", "just what i needed",
            "oh great", "wow", "bravo", "exactly what i wanted", "lucky me",
            "story of my life", "because that makes sense", "just my luck",
            "how convenient", "couldn't be better", "of course", "just perfect",
            "another", "one more", "fun times", "living the dream", "shocking",
            "not at all", "absolutely not", "no problem at all", "my favorite",
            "my hero", "way to go", "nice job", "nice work", "nice going", 
            "good job", "good work", "well done", "smooth move", "brilliant idea"
        ]
        
        # Check for patterns like "Oh great..." at the beginning of the text
        features['has_sarcastic_opening'] = any(text_lower.startswith(phrase) for phrase in ["oh great", "wow", "just what", "exactly what", "perfect", "fantastic"])
        
        # Improved sarcasm phrase detection (partial matching)
        features['has_sarcastic_phrases'] = any(phrase in text_lower for phrase in sarcastic_phrases)
        
        # Detect contrast between positive words and negative context
        positive_words = words & sentiment_patterns['positive']
        if positive_words and (features['has_negation'] or features['has_contrast']):
            features['has_positive_negative_contrast'] = True
        else:
            features['has_positive_negative_contrast'] = False
        
        # Check for sarcasm markers that indicate exaggerated sentiment
        features['has_exaggerated_sentiment'] = bool(
            re.search(r'(so|very|really|absolutely|totally|completely) (great|wonderful|perfect|fantastic|terrible|awful|horrible)', text_lower)
        )
        
        # Check for common sarcastic constructions
        features['has_sarcastic_construction'] = bool(
            re.search(r'(just|exactly|precisely|absolutely) what (i|we|you) (needed|wanted|expected|hoped for)', text_lower) or
            re.search(r'(because|cause) that (makes sense|works|helps)', text_lower) or
            re.search(r'(of course|naturally|obviously) .* (would|wouldn\'t|had to|didn\'t)', text_lower)
        )
        
        return features

    def preprocess_text(self, text):
        """Enhanced text preprocessing for better sarcasm detection"""
        text = text.lower().strip()
        text = ' '.join(text.split())
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        text = re.sub(r'(!+)', ' ! ', text)
        text = re.sub(r'(\?+)', ' ? ', text)
        text = re.sub(r'(\.+)', ' . ', text)

        # Add sarcasm indicators
        sarcasm_indicators = ['obviously', 'clearly', 'surely', 'totally', 'really', 'sarcastic']
        for indicator in sarcasm_indicators:
            if indicator in text:
                text += " [sarcasm_indicator]"

        return text

    def load_or_train_model(self):
        """Load or train the sarcasm detection model."""
        if not self.load_trained_model():
            print("No trained model found. Training new model...")
            self.train_model()
            self.save_model()

    def save_model(self):
        """Save the trained model and tokenizer."""
        print("Saving model...")
        model_dir = './models/sarcasm_model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save model
        torch.save(self.model.state_dict(), os.path.join(model_dir, 'model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_dir)
        print("Model saved successfully!")

    def load_trained_model(self):
        """Try to load a previously trained model."""
        try:
            model_path = os.path.join('./models/sarcasm_model', 'model.pt')
            if os.path.exists(model_path):
                print("Loading pre-trained model...")
                self.model.load_weights(model_path)
                
                # Try to load the tokenizer
                tokenizer_path = './models/sarcasm_model'
                if os.path.exists(tokenizer_path):
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                        print("Loaded saved tokenizer")
                    except Exception as e:
                        print(f"Could not load saved tokenizer: {e}")
                        print("Using original tokenizer")
                
                print("Model loaded successfully!")
                return True
            return False
        except Exception as e:
            print(f"Failed to load model: {e}")
            traceback.print_exc()
            return False

    def load_sarcasm_dataset(self):
        """Load sarcasm dataset from multiple CSV files."""
        dataset_paths = ['datasets/sarcasm.csv', 'datasets/sarcasm2.csv']
        combined_data = []

        try:
            for dataset_path in dataset_paths:
                if not os.path.exists(dataset_path):
                    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

                print(f"Loading dataset from {dataset_path}...")
                df = pd.read_csv(dataset_path)

                # Ensure the dataset has the expected columns
                if 'text' not in df.columns or 'is_sarcastic' not in df.columns:
                    raise ValueError(f"Dataset {dataset_path} must contain 'text' and 'is_sarcastic' columns.")

                combined_data.append(df[['text', 'is_sarcastic']])

            self.sarcasm_data = pd.concat(combined_data, ignore_index=True)
            self.sarcasm_data['is_sarcastic'] = self.sarcasm_data['is_sarcastic'].astype(int)
            print(f"Loaded {len(self.sarcasm_data)} examples from combined datasets.")

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            self.sarcasm_data = pd.DataFrame({
                'text': ['This is a sarcastic headline', 'This is a normal headline'],
                'is_sarcastic': [1, 0]
            })
            print("Using dummy dataset for testing")

    def train_model(self):
        try:
            print("Starting model training...")
            if len(self.sarcasm_data) == 0:
                raise ValueError("No training data available")

            print("\nDataset Statistics:")
            print(f"Total examples: {len(self.sarcasm_data)}")
            print("Label distribution:")
            print(self.sarcasm_data['is_sarcastic'].value_counts())
            print("\n")

            train_texts, test_texts, train_labels, test_labels = train_test_split(
                self.sarcasm_data['text'].tolist(),
                self.sarcasm_data['is_sarcastic'].tolist(),
                test_size=0.2,
                random_state=42,
                stratify=self.sarcasm_data['is_sarcastic']
            )

            print(f"Training on {len(train_texts)} examples, testing on {len(test_texts)} examples")

            # Create a simple training loop without Hugging Face Trainer
            # This avoids the rotary embedding assertion error
            train_input_ids = []
            train_attention_masks = []
            for text in train_texts:
                encoded = self.tokenizer.encode_plus(
                    text,
                max_length=128,
                truncation=True, 
                    padding='max_length',
                return_tensors='pt'
            )
                train_input_ids.append(encoded['input_ids'])
                train_attention_masks.append(encoded['attention_mask'])
            
            train_input_ids = torch.cat(train_input_ids, dim=0)
            train_attention_masks = torch.cat(train_attention_masks, dim=0)
            train_labels = torch.tensor(train_labels)
            
            # Create dataset
            train_dataset = torch.utils.data.TensorDataset(
                train_input_ids, 
                train_attention_masks, 
                train_labels
            )
            
            # Use smaller batch size
            batch_size = 8  # Reduced from 16
            
            # Create data loader with smaller batch size
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Create optimizer with memory efficient settings
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=2e-5,
                weight_decay=0.01,
                eps=1e-8
            )
            
            criterion = torch.nn.CrossEntropyLoss()
            
            print("\nTraining model with mixed precision...")
            num_epochs = 3
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                
                for batch in train_loader:
                    # Clear memory
                    torch.cuda.empty_cache()
                    
                    optimizer.zero_grad()
                    
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)
                    
                    # Use automatic mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model.forward_no_rotary(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                    
                    # Scale loss and backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    total_loss += loss.item()
                    
                    # Clear memory after each batch
                    del loss, outputs
                    torch.cuda.empty_cache()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            print("\nModel training complete!")
            self.save_model()

        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            self.save_model()  # Save even if there's an error

    def calculate_feature_score(self, features):
        """Calculate a sarcasm score based on extracted features with improved accuracy."""
        # Base score - start lower to create stronger bias toward sincere
        score = 0.38
        
        # Strong sarcasm indicators
        if features.get('has_sarcastic_phrases'):
            score += 0.35
        if features.get('has_mock_praise') and features.get('has_negative_context', False):
            score += 0.30  # Only count mock praise if in negative context
        if features.get('has_sentiment_contrast'):
            score += 0.25
        if features.get('has_sarcastic_opening'):
            score += 0.30
        if features.get('has_positive_negative_contrast'):
            score += 0.28
        if features.get('has_sarcastic_construction'):
            score += 0.35
        if features.get('has_exaggerated_sentiment'):
            score += 0.25
        if features.get('has_exaggeration'):
            score += 0.20
        
        # Moderate sarcasm indicators
        if features.get('has_rhetorical_question'):
            score += 0.18
        if features.get('has_punctuation_emphasis') and (
            features.get('has_mock_praise') or 
            features.get('has_contrast') or 
            features.get('has_sentiment_contrast')
        ):
            score += 0.18  # Only count emphasis if with other indicators
        
        # Improved contrast detection
        if features.get('has_contrast'):
            text_lower = self.last_analyzed_text.lower() if hasattr(self, 'last_analyzed_text') else ""
            
            # Expanded sincere context patterns
            sincere_context = [
                'positive', 'good', 'better', 'improvement', 'hope', 'fresh', 'new',
                'part of', 'have to', 'need to', 'requires', 'necessary', 'job',
                'work', 'responsibility', 'important', 'simple', 'process', 'slow',
                'challenging', 'difficult', 'tough', 'hard', 'help', 'learn',
                'balance', 'practice', 'routine', 'habit', 'skill', 'progress',
                'growth', 'development', 'improvement', 'achievement', 'success',
                'effort', 'discipline', 'persistence', 'patience', 'mindfulness'
            ]
            
            # Common sincere phrases with contrasts
            sincere_phrases = [
                "but it's part of", "but it works", "but it helps", 
                "tough but", "hard but", "slow but", "difficult but",
                "but i still", "but we still", "but they still",
                "but that's okay", "but that's fine", "but it's worth",
                "challenging but rewarding", "difficult but possible",
                "hard work but", "takes time but", "expensive but worth",
                "old but reliable", "simple but effective", "basic but functional"
            ]
            
            # If contrast appears with sincere words or phrases, it's likely not sarcastic
            has_sincere_context = any(word in text_lower for word in sincere_context)
            has_sincere_phrase = any(phrase in text_lower for phrase in sincere_phrases)
            
            if has_sincere_phrase:
                # Strong reduction for obvious sincere contrast phrases
                score -= 0.25
            elif has_sincere_context:
                # Moderate reduction for general sincere context
                score -= 0.20
        else:
                # Otherwise small boost for contrast
                score += 0.12
        
        # Weak sarcasm indicators
        if features.get('has_capitalization'):
            score += 0.08
        if features.get('has_quotes'):
            score += 0.05
        
        # Special case: if text is a straightforward statement about work/process
        # with contrasts but no strong sarcasm markers, likely sincere
        text_lower = self.last_analyzed_text.lower() if hasattr(self, 'last_analyzed_text') else ""
        work_process_words = [
            'work', 'job', 'process', 'task', 'duty', 'project', 'responsibility',
            'function', 'operation', 'procedure', 'method', 'technique', 'approach',
            'routine', 'practice', 'habit', 'schedule', 'plan', 'strategy'
        ]
        
        if any(word in text_lower for word in work_process_words) and features.get('has_contrast'):
            if not (features.get('has_sarcastic_phrases') or features.get('has_punctuation_emphasis')):
                # Likely sincere work-related statement
                score -= 0.18
        
        # Normalize score between 0 and 1
        score = min(max(score, 0), 1)
        
        return score

    def _generate_sarcasm_explanation(self, confidence, features, text):
        """Generate enhanced explanation for sarcastic text with specific markers."""
        if confidence > 0.85:
            base = "This is highly sarcastic! "
        elif confidence > 0.65:
            base = "This appears to be sarcastic. "
        else:
            base = "This might contain sarcasm. "
        
        # Extract the most relevant features for explanation
        reasons = []
        
        # Add specific pattern detection
        text_lower = text.lower()
        
        # Look for specific patterns to highlight in the explanation
        if features.get('has_sarcastic_phrases'):
            sarcastic_phrases = [
                "yeah right", "oh really", "sure", "totally", "as if", 
                "what a surprise", "no way", "you don't say", "just what i needed",
                "oh great", "wow", "bravo", "lucky me", "story of my life",
                "because that makes sense", "just my luck", "how convenient",
                "couldn't be better", "of course", "just perfect"
            ]
            found_phrases = [phrase for phrase in sarcastic_phrases if phrase in text_lower]
            if found_phrases:
                phrase_str = f'"{found_phrases[0]}"'
                reasons.append(f"it contains the sarcastic phrase {phrase_str}")
        
        if features.get('has_mock_praise') and features.get('has_negative_context', False):
            reasons.append("it includes exaggerated praise in a negative context")
        elif features.get('has_mock_praise'):
            reasons.append("it includes exaggerated praise")
        
        if features.get('has_sentiment_contrast'):
            reasons.append("it mixes positive and negative sentiments")
        
        if features.get('has_punctuation_emphasis'):
            if "!?" in text:
                reasons.append("it uses emotional punctuation (!?)")
            elif "!!" in text:
                reasons.append("it uses excessive exclamation marks")
            elif "..." in text:
                reasons.append("it uses ellipses to create dramatic effect")
            
        if features.get('has_exaggeration'):
            reasons.append("it uses exaggerated terms like 'always' or 'never'")
        
        if features.get('has_rhetorical_question'):
            reasons.append("it contains a rhetorical question")
        
        if features.get('has_sarcastic_construction') and not any(reasons):
            reasons.append("it uses a common sarcastic construction")
        
        if features.get('has_contrasting_statements') and not any(reasons):
            reasons.append("it contains contrasting statements")
        
        if features.get('has_capitalization') and not any(reasons):
            reasons.append("it uses CAPITALIZATION for emphasis")
        
        # Add a catch-all if no specific reasons were identified
        if not reasons:
            if features.get('has_contrast'):
                reasons.append("it contains contrasting statements")
            else:
                reasons.append("the language patterns suggest sarcasm")
            
        return base + "I detected this because " + ", ".join(reasons) + "."

    def _generate_sincere_explanation(self, confidence, text):
        """Generate enhanced explanation for sincere text."""
        text_lower = text.lower()
        
        # Look for specific sincere patterns to explain the classification
        if confidence > 0.85:
            base = "This appears to be completely sincere. "
            
            # Add specific explanations for very confident cases
            sincere_explanations = []
            
            # Check for factual statements
            factual_markers = ["is located in", "was founded in", "consists of", "is made of", "contains"]
            if any(marker in text_lower for marker in factual_markers):
                sincere_explanations.append("It appears to be a factual statement")
            
            # Check for educational content
            educational_markers = ["according to", "research shows", "studies have found", "historically"]
            if any(marker in text_lower for marker in educational_markers):
                sincere_explanations.append("It contains educational/informative content")
            
            # Check for straightforward positive sentiment
            positive_markers = ["happy to", "glad to", "pleased to", "looking forward to", "excited to"]
            if any(marker in text_lower for marker in positive_markers):
                sincere_explanations.append("It expresses genuine positive sentiment")
            
            # Add a default explanation if none of the above apply
            if not sincere_explanations:
                return base + "I didn't detect any typical sarcastic patterns."
            
            return base + sincere_explanations[0] + "."
            
        elif confidence > 0.65:
            return "This is likely sincere. Few or no sarcastic indicators were found."
        else:
            return "This is probably sincere, but I'm not entirely sure. The text lacks strong sarcastic markers."

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() 
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def calculate_clown_score(sentiment_score):
    # More nuanced calculation based on sentiment extremity
    neutral_point = 0.5
    extremity = abs(sentiment_score - neutral_point)
    
    # Higher score for more extreme sentiments
    base_score = extremity * 1.5
    
    # Add randomness factor to make it more interesting
    randomness = np.random.normal(0, 0.1)  # Small random adjustment
    
    # Ensure final score is between 0 and 1
    return min(max(base_score + randomness, 0), 1.0)

# Load sentiment analysis dataset
def load_sentiment_data():
    df = pd.read_csv('path/to/sentiment_dataset.csv')
    return train_test_split(df['text'], df['label'], test_size=0.2)

# Train a sentiment analysis model
def train_sentiment_model():
    train_texts, test_texts, train_labels, test_labels = load_sentiment_data()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

    train_dataset = SarcasmDataset(train_encodings, train_labels)
    test_dataset = SarcasmDataset(test_encodings, test_labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

class SarcasmNeuralNetwork(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Initialize the LLaMA model
        print("Initializing LLaMA model for sarcasm detection...")
        self.model = LlamaModel(params)
        
        # Add classification head for sarcasm detection
        self.classifier = nn.Linear(self.model.dim, 2)  # Binary classification: sarcastic or not
        
        print("Model initialized successfully!")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        # Get LLaMA model outputs
        outputs = self.model(input_ids)
        
        # Use the [CLS] token or mean of the sequence for classification
        pooled_output = outputs.mean(dim=1)
        
        # Classification head
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }

    def load_weights(self, checkpoint_path):
        """Load weights for fine-tuned sarcasm model."""
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded sarcasm model weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load sarcasm model weights: {e}")

class SarcasmTextProcessor:
    def __init__(self):
        # Load the tokenizer from local files
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(LLAMA_MODEL_PATH / "tokenizer.model")  # Ensure this points to the local tokenizer
        )

    def extract_contextual_features(self, text):
        """Enhanced contextual feature extraction with more nuanced patterns"""
        # Context patterns (expanded)
        context_patterns = {
            'contrast_markers': {
                'but', 'however', 'although', 'though', 'yet', 'nevertheless',
                'surprisingly', 'unexpectedly', 'ironically', 'funny how',
                'imagine that', 'shock', 'shocking', 'supposedly', 'apparently'
            },
            'emphasis_markers': {
                'very', 'so', 'such', 'quite', 'rather', 'absolutely', 'totally',
                'completely', 'utterly', 'literally', 'actually', 'seriously'
            },
            'mock_praise': {
                'genius', 'brilliant', 'outstanding', 'impressive', 'remarkable',
                'extraordinary', 'exceptional', 'magnificent', 'splendid'
            },
            'understatement': {
                'slightly', 'somewhat', 'maybe', 'perhaps', 'possibly',
                'presumably', 'supposedly', 'allegedly'
            }
        }
        
        # Expanded sentiment analysis
        sentiment_patterns = {
            'positive': {
                'great', 'wonderful', 'amazing', 'fantastic', 'brilliant',
                'love', 'awesome', 'excellent', 'perfect', 'delightful',
                'outstanding', 'incredible', 'superb', 'magnificent'
            },
            'negative': {
                'worst', 'terrible', 'awful', 'horrible', 'disaster',
                'hate', 'boring', 'pathetic', 'ridiculous', 'useless',
                'disappointing', 'mediocre', 'failure', 'mess'
            }
        }

        text_lower = text.lower()
        words = set(text_lower.split())
        sentences = text.split('.')
        
        # Enhanced feature extraction
        features = {
            'has_contrast': any(marker in text_lower for marker in context_patterns['contrast_markers']),
            'has_emphasis': any(marker in text_lower for marker in context_patterns['emphasis_markers']),
            'has_mock_praise': any(marker in text_lower for marker in context_patterns['mock_praise']),
            'has_understatement': any(marker in text_lower for marker in context_patterns['understatement']),
            'has_sentiment_contrast': bool(words & sentiment_patterns['positive']) and bool(words & sentiment_patterns['negative']),
            'has_punctuation_emphasis': '!!' in text or '??' in text or '!?' in text or '...' in text,
            'has_quotes': '"' in text or "'" in text,
            'has_capitalization': any(word.isupper() for word in text.split()),
            'sentence_length': len(sentences),
            'has_negation': any(neg in text_lower for neg in ['not', 'never', 'no', 'none']),
            'has_rhetorical_question': '?' in text and any(q.strip().startswith(('why', 'how', 'what', 'really', 'seriously')) for q in text.split('?')),
            'repeated_words': any(text_lower.split().count(word.lower()) > 1 for word in words),
            'has_negative_context': bool(words & sentiment_patterns['negative']) or any(
                neg in text_lower for neg in ['problem', 'issue', 'trouble', 'difficult', 'hard', 'challenge']
            ),
        }
        
        # Check for common sarcastic phrases
        sarcastic_phrases = [
            "yeah right", "oh really", "sure", "totally", "as if", 
            "what a surprise", "no way", "you don't say", "just what I needed"
        ]
        features['has_sarcastic_phrases'] = any(phrase in text_lower for phrase in sarcastic_phrases)
        
        return features

    def preprocess_text(self, text):
        # Preprocess the text for the model
        return self.tokenizer(text, return_tensors='pt', truncation=True, padding=True) 