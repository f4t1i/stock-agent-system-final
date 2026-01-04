"""
Technical Analysis Agent - Junior Agent für technische Indikatoren
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..base_agent import BaseAgent


class TechnicalAgent(BaseAgent):
    """
    Spezialisierter Agent für technische Analyse.
    
    Kombiniert deterministische Indikatorberechnung mit LLM-basierter Interpretation.
    """
    
    def __init__(self, model_path: str, config: Dict):
        super().__init__(config)
        self.model_path = model_path
        self.config = config
        
        # Load model für Interpretation
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if config.get('fp16', True) else torch.float32,
            device_map="auto"
        )
        
        if config.get('lora_adapter_path'):
            self.model = PeftModel.from_pretrained(
                self.model,
                config['lora_adapter_path']
            )
        
        self.model.eval()
        
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """System-Prompt für Technical-Agent"""
        return """You are an expert technical analyst. Your task is to:

1. Interpret calculated technical indicators
2. Identify chart patterns and trends
3. Assess momentum and potential reversals
4. Determine support/resistance levels
5. Provide actionable trading signals

Always provide:
- Overall trend assessment
- Momentum evaluation
- Key support/resistance levels
- Signal strength (0 to 1)
- Recommended action (buy/sell/hold)
- Risk assessment

Output must be valid JSON following this schema:
{
    "trend": "uptrend" | "downtrend" | "sideways",
    "momentum": "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish",
    "signal": "buy" | "sell" | "hold",
    "signal_strength": float,
    "support_levels": [float],
    "resistance_levels": [float],
    "reasoning": string,
    "risk_assessment": string,
    "key_patterns": [string]
}
"""
    
    def analyze(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> Dict:
        """
        Technische Analyse für ein Symbol
        
        Args:
            symbol: Stock symbol
            period: Zeitraum (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Intervall (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dict mit technischer Analyse
        """
        # Lade Marktdaten
        df = self._fetch_market_data(symbol, period, interval)
        
        if df is None or len(df) < 50:  # Mindestens 50 Datenpunkte
            return {
                'error': 'Insufficient data for technical analysis',
                'symbol': symbol
            }
        
        # Berechne Indikatoren
        indicators = self._calculate_indicators(df)
        
        # Identifiziere Patterns
        patterns = self._identify_patterns(df)
        
        # Support/Resistance
        support, resistance = self._find_support_resistance(df)
        
        # Format für LLM
        technical_summary = self._format_technical_summary(
            df, indicators, patterns, support, resistance
        )
        
        # LLM-Interpretation
        interpretation = self._interpret_technicals(symbol, technical_summary)
        
        # Kombiniere deterministische + LLM-Ergebnisse
        result = {
            **interpretation,
            'indicators': indicators,
            'patterns': patterns,
            'current_price': float(df['Close'].iloc[-1]),
            'metadata': {
                'period': period,
                'interval': interval,
                'data_points': len(df),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return result
    
    def _fetch_market_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Lade Marktdaten von Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Berechne technische Indikatoren"""
        
        # Trend-Indikatoren
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.ema(length=12, append=True)
        df.ta.ema(length=26, append=True)
        
        # Momentum
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(append=True)
        df.ta.macd(append=True)
        
        # Volatilität
        df.ta.bbands(length=20, append=True)
        df.ta.atr(length=14, append=True)
        
        # Volume
        df.ta.obv(append=True)
        df.ta.mfi(append=True)
        
        # Extrahiere letzte Werte
        latest = df.iloc[-1]
        
        indicators = {
            # Trend
            'sma_20': float(latest.get('SMA_20', 0)),
            'sma_50': float(latest.get('SMA_50', 0)),
            'sma_200': float(latest.get('SMA_200', 0)),
            'ema_12': float(latest.get('EMA_12', 0)),
            'ema_26': float(latest.get('EMA_26', 0)),
            
            # Momentum
            'rsi_14': float(latest.get('RSI_14', 50)),
            'stoch_k': float(latest.get('STOCHk_14_3_3', 50)),
            'stoch_d': float(latest.get('STOCHd_14_3_3', 50)),
            'macd': float(latest.get('MACD_12_26_9', 0)),
            'macd_signal': float(latest.get('MACDs_12_26_9', 0)),
            'macd_histogram': float(latest.get('MACDh_12_26_9', 0)),
            
            # Bollinger Bands
            'bb_upper': float(latest.get('BBU_20_2.0', 0)),
            'bb_middle': float(latest.get('BBM_20_2.0', 0)),
            'bb_lower': float(latest.get('BBL_20_2.0', 0)),
            
            # Volume
            'obv': float(latest.get('OBV', 0)),
            'mfi': float(latest.get('MFI_14', 50)),
            
            # Volatilität
            'atr': float(latest.get('ATR_14', 0))
        }
        
        # Abgeleitete Signale
        current_price = float(df['Close'].iloc[-1])
        
        # Trend-Bewertung
        if current_price > indicators['sma_20'] > indicators['sma_50']:
            indicators['trend_signal'] = 'bullish'
        elif current_price < indicators['sma_20'] < indicators['sma_50']:
            indicators['trend_signal'] = 'bearish'
        else:
            indicators['trend_signal'] = 'neutral'
        
        # Momentum-Bewertung
        rsi = indicators['rsi_14']
        if rsi > 70:
            indicators['momentum_signal'] = 'overbought'
        elif rsi < 30:
            indicators['momentum_signal'] = 'oversold'
        else:
            indicators['momentum_signal'] = 'neutral'
        
        # MACD-Signal
        if indicators['macd'] > indicators['macd_signal']:
            indicators['macd_signal'] = 'bullish'
        else:
            indicators['macd_signal'] = 'bearish'
        
        # Bollinger Band Position
        bb_range = indicators['bb_upper'] - indicators['bb_lower']
        if bb_range > 0:
            bb_position = (current_price - indicators['bb_lower']) / bb_range
            if bb_position > 0.8:
                indicators['bb_position'] = 'upper'
            elif bb_position < 0.2:
                indicators['bb_position'] = 'lower'
            else:
                indicators['bb_position'] = 'middle'
        else:
            indicators['bb_position'] = 'middle'
        
        return indicators
    
    def _identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identifiziere Chart-Patterns"""
        patterns = []
        
        # Golden/Death Cross
        if len(df) >= 200:
            sma_50 = df['Close'].rolling(50).mean()
            sma_200 = df['Close'].rolling(200).mean()
            
            if sma_50.iloc[-2] < sma_200.iloc[-2] and sma_50.iloc[-1] > sma_200.iloc[-1]:
                patterns.append("Golden Cross (SMA50 crossed above SMA200)")
            elif sma_50.iloc[-2] > sma_200.iloc[-2] and sma_50.iloc[-1] < sma_200.iloc[-1]:
                patterns.append("Death Cross (SMA50 crossed below SMA200)")
        
        # Higher Highs / Lower Lows
        recent_highs = df['High'].tail(20)
        recent_lows = df['Low'].tail(20)
        
        if recent_highs.iloc[-1] > recent_highs.iloc[-2] > recent_highs.iloc[-3]:
            patterns.append("Higher Highs (Uptrend)")
        elif recent_lows.iloc[-1] < recent_lows.iloc[-2] < recent_lows.iloc[-3]:
            patterns.append("Lower Lows (Downtrend)")
        
        # Volume Spike
        avg_volume = df['Volume'].rolling(20).mean()
        if df['Volume'].iloc[-1] > 1.5 * avg_volume.iloc[-1]:
            patterns.append("Volume Spike")
        
        return patterns
    
    def _find_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> tuple[List[float], List[float]]:
        """Identifiziere Support- und Resistance-Levels"""
        
        # Lokale Minima = Support
        # Lokale Maxima = Resistance
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(df) - window):
            # Support: Lokales Minimum
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
                support_levels.append(float(df['Low'].iloc[i]))
            
            # Resistance: Lokales Maximum
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
                resistance_levels.append(float(df['High'].iloc[i]))
        
        # Clustere nahe Levels
        support_levels = self._cluster_levels(support_levels)
        resistance_levels = self._cluster_levels(resistance_levels)
        
        # Nur relevante Levels (nahe aktuellem Preis)
        current_price = float(df['Close'].iloc[-1])
        
        support_levels = [
            s for s in support_levels 
            if current_price * 0.90 <= s <= current_price
        ][-3:]  # Top 3
        
        resistance_levels = [
            r for r in resistance_levels 
            if current_price <= r <= current_price * 1.10
        ][:3]  # Top 3
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Clustere nahe Price-Levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] < current_cluster[-1] * threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _format_technical_summary(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        patterns: List[str],
        support: List[float],
        resistance: List[float]
    ) -> str:
        """Formatiere technische Daten für LLM"""
        
        current_price = df['Close'].iloc[-1]
        
        summary = f"""Current Price: ${current_price:.2f}

TREND INDICATORS:
- SMA 20: ${indicators['sma_20']:.2f} ({('Above' if current_price > indicators['sma_20'] else 'Below')})
- SMA 50: ${indicators['sma_50']:.2f} ({('Above' if current_price > indicators['sma_50'] else 'Below')})
- SMA 200: ${indicators['sma_200']:.2f} ({('Above' if current_price > indicators['sma_200'] else 'Below')})
- Trend Signal: {indicators['trend_signal']}

MOMENTUM INDICATORS:
- RSI (14): {indicators['rsi_14']:.1f} - {indicators['momentum_signal']}
- Stochastic K: {indicators['stoch_k']:.1f}
- Stochastic D: {indicators['stoch_d']:.1f}
- MACD: {indicators['macd']:.2f} (Signal: {indicators['macd_signal']})
- MACD Histogram: {indicators['macd_histogram']:.2f}

BOLLINGER BANDS:
- Upper: ${indicators['bb_upper']:.2f}
- Middle: ${indicators['bb_middle']:.2f}
- Lower: ${indicators['bb_lower']:.2f}
- Position: {indicators['bb_position']}

VOLUME:
- OBV: {indicators['obv']:.0f}
- MFI (14): {indicators['mfi']:.1f}

SUPPORT LEVELS: {', '.join([f'${s:.2f}' for s in support]) if support else 'None identified'}
RESISTANCE LEVELS: {', '.join([f'${r:.2f}' for r in resistance]) if resistance else 'None identified'}

IDENTIFIED PATTERNS:
{chr(10).join(['- ' + p for p in patterns]) if patterns else '- None'}
"""
        
        return summary
    
    def _interpret_technicals(self, symbol: str, technical_summary: str) -> Dict:
        """LLM-Interpretation der technischen Daten"""
        
        user_message = f"""Analyze the technical indicators for {symbol}:

{technical_summary}

Provide your interpretation in JSON format."""
        
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
                temperature=self.config.get('temperature', 0.5),
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse JSON
        try:
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            result = json.loads(response.strip())
            return result
            
        except Exception as e:
            # Fallback
            return {
                'trend': 'sideways',
                'momentum': 'neutral',
                'signal': 'hold',
                'signal_strength': 0.5,
                'support_levels': [],
                'resistance_levels': [],
                'reasoning': f'Error parsing response: {str(e)}',
                'risk_assessment': 'Unable to assess',
                'key_patterns': []
            }


if __name__ == "__main__":
    config = {
        'fp16': True,
        'max_new_tokens': 512,
        'temperature': 0.5
    }
    
    agent = TechnicalAgent(
        model_path="models/technical_agent_v1",
        config=config
    )
    
    result = agent.analyze("AAPL", period="3mo")
    print(json.dumps(result, indent=2))
