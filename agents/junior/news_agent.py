"""
News Sentiment Agent - Junior Agent für Sentiment-Analyse
"""

import json
from typing import Dict, List, Optional
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..base_agent import BaseAgent
from utils.news_fetcher import NewsFetcher


class NewsAgent(BaseAgent):
    """
    Spezialisierter Agent für News-Sentiment-Analyse.
    
    Trainiert via SFT auf synthetischen und annotierten News-Daten.
    """
    
    def __init__(self, model_path: str, config: Dict):
        super().__init__(config)
        self.model_path = model_path
        self.config = config
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if config.get('fp16', True) else torch.float32,
            device_map="auto"
        )
        
        # Load LoRA adapter if exists
        if config.get('lora_adapter_path'):
            self.model = PeftModel.from_pretrained(
                self.model,
                config['lora_adapter_path']
            )
        
        self.model.eval()
        
        # News fetcher
        self.news_fetcher = NewsFetcher(config.get('news_api_key'))
        
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Konstruiere System-Prompt für News-Agent"""
        return """You are a specialized financial news analyst. Your task is to:

1. Analyze news articles about a given stock
2. Extract key information that could impact stock price
3. Quantify the sentiment on a scale from -2 (very negative) to +2 (very positive)
4. Assess the likely price impact (bullish/bearish/neutral)
5. Determine the relevant time horizon (short_term/medium_term/long_term)

Always provide:
- A sentiment score (-2 to +2)
- Confidence level (0 to 1)
- Reasoning based on concrete facts
- Price impact assessment
- Time horizon

Output must be valid JSON following this schema:
{
    "sentiment_score": float,
    "confidence": float,
    "reasoning": string,
    "price_impact": "bullish" | "bearish" | "neutral",
    "time_horizon": "short_term" | "medium_term" | "long_term",
    "key_events": [string],
    "risk_factors": [string]
}
"""
    
    def analyze(
        self,
        symbol: str,
        news_articles: Optional[List[Dict]] = None,
        lookback_days: int = 7
    ) -> Dict:
        """
        Analysiere News-Sentiment für ein Symbol
        
        Args:
            symbol: Stock symbol (z.B. 'AAPL')
            news_articles: Optional liste von News-Artikeln. 
                          Wenn None, werden automatisch aktuelle News abgerufen.
            lookback_days: Anzahl Tage für News-Recherche
            
        Returns:
            Dict mit Sentiment-Analyse
        """
        # Fetch news if not provided
        if news_articles is None:
            news_articles = self.news_fetcher.get_news(symbol, days=lookback_days)
        
        if not news_articles:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'reasoning': 'No news articles found for analysis',
                'price_impact': 'neutral',
                'time_horizon': 'short_term',
                'key_events': [],
                'risk_factors': []
            }
        
        # Format input
        user_message = self._format_news_input(symbol, news_articles)
        
        # Generate analysis
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_message}
        ]
        
        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.get('max_new_tokens', 512),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse JSON response
        try:
            result = self._parse_response(response)
            
            # Add metadata
            result['metadata'] = {
                'num_articles': len(news_articles),
                'timestamp': datetime.now().isoformat(),
                'model': self.model_path
            }
            
            return result
            
        except Exception as e:
            # Fallback auf neutrale Einschätzung bei Parse-Fehler
            return {
                'sentiment_score': 0.0,
                'confidence': 0.3,
                'reasoning': f'Error parsing response: {str(e)}',
                'price_impact': 'neutral',
                'time_horizon': 'short_term',
                'key_events': [],
                'risk_factors': [],
                'error': str(e)
            }
    
    def _format_news_input(self, symbol: str, news_articles: List[Dict]) -> str:
        """Formatiere News-Artikel für Input"""
        articles_text = []
        
        for i, article in enumerate(news_articles[:20], 1):  # Max 20 Artikel
            articles_text.append(f"""
Article {i}:
Title: {article.get('title', 'N/A')}
Source: {article.get('source', 'N/A')}
Date: {article.get('publishedAt', 'N/A')}
Summary: {article.get('description', article.get('content', 'N/A')[:200])}
""")
        
        return f"""Analyze the sentiment and price impact for {symbol} based on these news articles:

{''.join(articles_text)}

Provide a comprehensive sentiment analysis in JSON format."""
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON"""
        # Versuche JSON zu extrahieren
        # Manchmal wrapped das Modell JSON in Markdown-Code-Blocks
        
        # Entferne Markdown code blocks
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        # Parse JSON
        result = json.loads(response.strip())
        
        # Validiere erforderliche Felder
        required_fields = [
            'sentiment_score',
            'confidence',
            'reasoning',
            'price_impact',
            'time_horizon'
        ]
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Clamp sentiment score
        result['sentiment_score'] = max(-2.0, min(2.0, result['sentiment_score']))
        result['confidence'] = max(0.0, min(1.0, result['confidence']))
        
        return result
    
    def batch_analyze(self, symbols: List[str], lookback_days: int = 7) -> Dict[str, Dict]:
        """Batch-Analyse für mehrere Symbole"""
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol, lookback_days=lookback_days)
            except Exception as e:
                results[symbol] = {'error': str(e)}
        
        return results


# Beispiel-Nutzung
if __name__ == "__main__":
    config = {
        'fp16': True,
        'max_new_tokens': 512,
        'temperature': 0.7,
        'news_api_key': 'your_api_key'
    }
    
    agent = NewsAgent(
        model_path="models/news_agent_v1",
        config=config
    )
    
    result = agent.analyze("AAPL")
    print(json.dumps(result, indent=2))
