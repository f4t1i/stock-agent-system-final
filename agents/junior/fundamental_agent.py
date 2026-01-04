"""
Fundamental Analysis Agent - Junior Agent für Fundamentalanalyse
"""

import json
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import yfinance as yf

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..base_agent import BaseAgent


class FundamentalAgent(BaseAgent):
    """
    Spezialisierter Agent für Fundamentalanalyse.

    Analysiert Unternehmenskennzahlen, Finanzberichte und bewertet
    die fundamentale Stärke eines Unternehmens.
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
        """System-Prompt für Fundamental-Agent"""
        return """You are an expert fundamental analyst specializing in corporate finance and valuation. Your task is to:

1. Analyze financial statements and key metrics
2. Assess company valuation (P/E, P/B, EV/EBITDA, etc.)
3. Evaluate profitability (ROE, ROA, margins)
4. Analyze growth trends (revenue, earnings, EPS)
5. Assess financial health (debt levels, liquidity)
6. Compare metrics to industry peers and historical averages

Always provide:
- Valuation assessment (undervalued/fairly valued/overvalued)
- Financial health rating (0 to 1)
- Growth quality score (0 to 1)
- Investment recommendation (strong buy/buy/hold/sell/strong sell)
- Confidence level (0 to 1)
- Detailed reasoning based on metrics
- Key risks and opportunities

Output must be valid JSON following this schema:
{
    "valuation": "undervalued" | "fairly_valued" | "overvalued",
    "financial_health": float,
    "growth_quality": float,
    "recommendation": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
    "confidence": float,
    "reasoning": string,
    "key_strengths": [string],
    "key_risks": [string],
    "price_target": float | null,
    "time_horizon": "short_term" | "medium_term" | "long_term"
}
"""

    def analyze(
        self,
        symbol: str,
        period: str = "Q"
    ) -> Dict:
        """
        Fundamentalanalyse für ein Symbol

        Args:
            symbol: Stock symbol (z.B. 'AAPL')
            period: 'Q' für Quartal, 'Y' für Jahr

        Returns:
            Dict mit Fundamentalanalyse
        """
        # Lade Unternehmensdaten
        company_info = self._fetch_company_info(symbol)

        if not company_info:
            return {
                'error': 'Unable to fetch company information',
                'symbol': symbol
            }

        # Berechne Kennzahlen
        metrics = self._calculate_metrics(symbol, company_info)

        if not metrics:
            return {
                'error': 'Unable to calculate fundamental metrics',
                'symbol': symbol
            }

        # Format für LLM
        fundamental_summary = self._format_fundamental_summary(
            symbol, company_info, metrics
        )

        # LLM-Interpretation
        interpretation = self._interpret_fundamentals(symbol, fundamental_summary)

        # Kombiniere Metriken + Interpretation
        result = {
            **interpretation,
            'metrics': metrics,
            'company_info': {
                'sector': company_info.get('sector', 'N/A'),
                'industry': company_info.get('industry', 'N/A'),
                'market_cap': company_info.get('marketCap', 0),
                'employees': company_info.get('fullTimeEmployees', 0)
            },
            'metadata': {
                'period': period,
                'timestamp': datetime.now().isoformat(),
                'model': self.model_path
            }
        }

        return result

    def _fetch_company_info(self, symbol: str) -> Optional[Dict]:
        """Lade Unternehmensinformationen"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'symbol' not in info:
                return None

            return info

        except Exception as e:
            print(f"Error fetching company info for {symbol}: {e}")
            return None

    def _calculate_metrics(self, symbol: str, info: Dict) -> Optional[Dict]:
        """Berechne fundamentale Kennzahlen"""

        try:
            ticker = yf.Ticker(symbol)

            # Valuation Metrics
            pe_ratio = info.get('trailingPE', None)
            forward_pe = info.get('forwardPE', None)
            pb_ratio = info.get('priceToBook', None)
            ps_ratio = info.get('priceToSalesTrailing12Months', None)
            ev_to_ebitda = info.get('enterpriseToEbitda', None)
            peg_ratio = info.get('pegRatio', None)

            # Profitability Metrics
            profit_margin = info.get('profitMargins', None)
            operating_margin = info.get('operatingMargins', None)
            roe = info.get('returnOnEquity', None)
            roa = info.get('returnOnAssets', None)

            # Growth Metrics
            revenue_growth = info.get('revenueGrowth', None)
            earnings_growth = info.get('earningsGrowth', None)
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', None)

            # Financial Health
            current_ratio = info.get('currentRatio', None)
            quick_ratio = info.get('quickRatio', None)
            debt_to_equity = info.get('debtToEquity', None)
            total_debt = info.get('totalDebt', 0)
            total_cash = info.get('totalCash', 0)
            free_cash_flow = info.get('freeCashflow', None)

            # Per Share Metrics
            eps = info.get('trailingEps', None)
            forward_eps = info.get('forwardEps', None)
            book_value = info.get('bookValue', None)
            revenue_per_share = info.get('revenuePerShare', None)

            # Dividend Info
            dividend_rate = info.get('dividendRate', None)
            dividend_yield = info.get('dividendYield', None)
            payout_ratio = info.get('payoutRatio', None)

            metrics = {
                # Valuation
                'pe_ratio': self._safe_float(pe_ratio),
                'forward_pe': self._safe_float(forward_pe),
                'pb_ratio': self._safe_float(pb_ratio),
                'ps_ratio': self._safe_float(ps_ratio),
                'ev_to_ebitda': self._safe_float(ev_to_ebitda),
                'peg_ratio': self._safe_float(peg_ratio),

                # Profitability
                'profit_margin': self._safe_float(profit_margin, multiply=100),
                'operating_margin': self._safe_float(operating_margin, multiply=100),
                'roe': self._safe_float(roe, multiply=100),
                'roa': self._safe_float(roa, multiply=100),

                # Growth
                'revenue_growth': self._safe_float(revenue_growth, multiply=100),
                'earnings_growth': self._safe_float(earnings_growth, multiply=100),
                'earnings_quarterly_growth': self._safe_float(earnings_quarterly_growth, multiply=100),

                # Financial Health
                'current_ratio': self._safe_float(current_ratio),
                'quick_ratio': self._safe_float(quick_ratio),
                'debt_to_equity': self._safe_float(debt_to_equity),
                'total_debt': total_debt,
                'total_cash': total_cash,
                'net_cash': total_cash - total_debt,
                'free_cash_flow': self._safe_float(free_cash_flow),

                # Per Share
                'eps': self._safe_float(eps),
                'forward_eps': self._safe_float(forward_eps),
                'book_value': self._safe_float(book_value),
                'revenue_per_share': self._safe_float(revenue_per_share),

                # Dividend
                'dividend_rate': self._safe_float(dividend_rate),
                'dividend_yield': self._safe_float(dividend_yield, multiply=100),
                'payout_ratio': self._safe_float(payout_ratio, multiply=100)
            }

            # Berechne abgeleitete Scores
            metrics['valuation_score'] = self._calculate_valuation_score(metrics)
            metrics['profitability_score'] = self._calculate_profitability_score(metrics)
            metrics['growth_score'] = self._calculate_growth_score(metrics)
            metrics['health_score'] = self._calculate_health_score(metrics)

            return metrics

        except Exception as e:
            print(f"Error calculating metrics for {symbol}: {e}")
            return None

    def _safe_float(self, value, multiply: float = 1.0) -> Optional[float]:
        """Sichere Float-Konvertierung"""
        if value is None or value == 'N/A':
            return None
        try:
            return float(value) * multiply
        except (ValueError, TypeError):
            return None

    def _calculate_valuation_score(self, metrics: Dict) -> float:
        """Berechne Valuation Score (0-1, höher = günstiger)"""
        score = 0.5  # Neutral start
        count = 0

        # P/E Ratio (lower is better, benchmark: 15)
        if metrics['pe_ratio']:
            if metrics['pe_ratio'] < 15:
                score += 0.1
            elif metrics['pe_ratio'] > 25:
                score -= 0.1
            count += 1

        # P/B Ratio (lower is better, benchmark: 3)
        if metrics['pb_ratio']:
            if metrics['pb_ratio'] < 3:
                score += 0.1
            elif metrics['pb_ratio'] > 5:
                score -= 0.1
            count += 1

        # PEG Ratio (lower is better, benchmark: 1)
        if metrics['peg_ratio']:
            if metrics['peg_ratio'] < 1:
                score += 0.15
            elif metrics['peg_ratio'] > 2:
                score -= 0.15
            count += 1

        return max(0.0, min(1.0, score))

    def _calculate_profitability_score(self, metrics: Dict) -> float:
        """Berechne Profitability Score (0-1)"""
        score = 0.0
        count = 0

        # ROE (benchmark: 15%)
        if metrics['roe']:
            if metrics['roe'] > 20:
                score += 0.25
            elif metrics['roe'] > 15:
                score += 0.15
            elif metrics['roe'] > 10:
                score += 0.05
            count += 1

        # ROA (benchmark: 5%)
        if metrics['roa']:
            if metrics['roa'] > 10:
                score += 0.25
            elif metrics['roa'] > 5:
                score += 0.15
            count += 1

        # Profit Margin (benchmark: 10%)
        if metrics['profit_margin']:
            if metrics['profit_margin'] > 20:
                score += 0.25
            elif metrics['profit_margin'] > 10:
                score += 0.15
            count += 1

        # Operating Margin
        if metrics['operating_margin']:
            if metrics['operating_margin'] > 15:
                score += 0.25
            elif metrics['operating_margin'] > 10:
                score += 0.15
            count += 1

        return score / max(count, 1) if count > 0 else 0.5

    def _calculate_growth_score(self, metrics: Dict) -> float:
        """Berechne Growth Score (0-1)"""
        score = 0.0
        count = 0

        # Revenue Growth
        if metrics['revenue_growth']:
            if metrics['revenue_growth'] > 20:
                score += 0.35
            elif metrics['revenue_growth'] > 10:
                score += 0.20
            elif metrics['revenue_growth'] > 5:
                score += 0.10
            count += 1

        # Earnings Growth
        if metrics['earnings_growth']:
            if metrics['earnings_growth'] > 20:
                score += 0.35
            elif metrics['earnings_growth'] > 10:
                score += 0.20
            count += 1

        # Quarterly Earnings Growth
        if metrics['earnings_quarterly_growth']:
            if metrics['earnings_quarterly_growth'] > 15:
                score += 0.30
            elif metrics['earnings_quarterly_growth'] > 5:
                score += 0.15
            count += 1

        return score / max(count, 1) if count > 0 else 0.5

    def _calculate_health_score(self, metrics: Dict) -> float:
        """Berechne Financial Health Score (0-1)"""
        score = 0.0
        count = 0

        # Current Ratio (benchmark: >2)
        if metrics['current_ratio']:
            if metrics['current_ratio'] > 2:
                score += 0.25
            elif metrics['current_ratio'] > 1.5:
                score += 0.15
            elif metrics['current_ratio'] < 1:
                score -= 0.15
            count += 1

        # Debt to Equity (lower is better, benchmark: <0.5)
        if metrics['debt_to_equity'] is not None:
            if metrics['debt_to_equity'] < 0.3:
                score += 0.25
            elif metrics['debt_to_equity'] < 0.5:
                score += 0.15
            elif metrics['debt_to_equity'] > 1.5:
                score -= 0.20
            count += 1

        # Free Cash Flow (positive is good)
        if metrics['free_cash_flow']:
            if metrics['free_cash_flow'] > 0:
                score += 0.25
            else:
                score -= 0.15
            count += 1

        # Net Cash (positive is better)
        if metrics['net_cash'] > 0:
            score += 0.25
        elif metrics['net_cash'] < 0:
            score -= 0.10
        count += 1

        return max(0.0, min(1.0, score / max(count, 1) + 0.5)) if count > 0 else 0.5

    def _format_fundamental_summary(
        self,
        symbol: str,
        info: Dict,
        metrics: Dict
    ) -> str:
        """Formatiere fundamentale Daten für LLM"""

        def fmt(val, suffix="", prefix=""):
            if val is None:
                return "N/A"
            return f"{prefix}{val:.2f}{suffix}"

        summary = f"""Company: {info.get('longName', symbol)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Market Cap: ${info.get('marketCap', 0):,.0f}

VALUATION METRICS:
- P/E Ratio (TTM): {fmt(metrics['pe_ratio'])}
- Forward P/E: {fmt(metrics['forward_pe'])}
- P/B Ratio: {fmt(metrics['pb_ratio'])}
- P/S Ratio: {fmt(metrics['ps_ratio'])}
- EV/EBITDA: {fmt(metrics['ev_to_ebitda'])}
- PEG Ratio: {fmt(metrics['peg_ratio'])}
- Valuation Score: {fmt(metrics['valuation_score'])}

PROFITABILITY METRICS:
- Profit Margin: {fmt(metrics['profit_margin'], '%')}
- Operating Margin: {fmt(metrics['operating_margin'], '%')}
- ROE: {fmt(metrics['roe'], '%')}
- ROA: {fmt(metrics['roa'], '%')}
- Profitability Score: {fmt(metrics['profitability_score'])}

GROWTH METRICS:
- Revenue Growth (YoY): {fmt(metrics['revenue_growth'], '%')}
- Earnings Growth (YoY): {fmt(metrics['earnings_growth'], '%')}
- Quarterly Earnings Growth: {fmt(metrics['earnings_quarterly_growth'], '%')}
- Growth Score: {fmt(metrics['growth_score'])}

FINANCIAL HEALTH:
- Current Ratio: {fmt(metrics['current_ratio'])}
- Quick Ratio: {fmt(metrics['quick_ratio'])}
- Debt/Equity: {fmt(metrics['debt_to_equity'])}
- Total Debt: ${metrics['total_debt']:,.0f}
- Total Cash: ${metrics['total_cash']:,.0f}
- Net Cash: ${metrics['net_cash']:,.0f}
- Free Cash Flow: {fmt(metrics['free_cash_flow'], prefix='$')}
- Health Score: {fmt(metrics['health_score'])}

PER SHARE METRICS:
- EPS (TTM): {fmt(metrics['eps'], prefix='$')}
- Forward EPS: {fmt(metrics['forward_eps'], prefix='$')}
- Book Value: {fmt(metrics['book_value'], prefix='$')}
- Revenue per Share: {fmt(metrics['revenue_per_share'], prefix='$')}

DIVIDEND INFO:
- Dividend Rate: {fmt(metrics['dividend_rate'], prefix='$')}
- Dividend Yield: {fmt(metrics['dividend_yield'], '%')}
- Payout Ratio: {fmt(metrics['payout_ratio'], '%')}
"""

        return summary

    def _interpret_fundamentals(self, symbol: str, fundamental_summary: str) -> Dict:
        """LLM-Interpretation der fundamentalen Daten"""

        user_message = f"""Analyze the fundamental metrics for {symbol}:

{fundamental_summary}

Provide your detailed fundamental analysis in JSON format."""

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
                temperature=self.config.get('temperature', 0.6),
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
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

            # Validate fields
            required_fields = [
                'valuation', 'financial_health', 'growth_quality',
                'recommendation', 'confidence', 'reasoning'
            ]

            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Clamp scores
            result['financial_health'] = max(0.0, min(1.0, result['financial_health']))
            result['growth_quality'] = max(0.0, min(1.0, result['growth_quality']))
            result['confidence'] = max(0.0, min(1.0, result['confidence']))

            return result

        except Exception as e:
            # Fallback
            return {
                'valuation': 'fairly_valued',
                'financial_health': 0.5,
                'growth_quality': 0.5,
                'recommendation': 'hold',
                'confidence': 0.3,
                'reasoning': f'Error parsing response: {str(e)}',
                'key_strengths': [],
                'key_risks': [],
                'price_target': None,
                'time_horizon': 'medium_term'
            }

    def batch_analyze(self, symbols: List[str], period: str = "Q") -> Dict[str, Dict]:
        """Batch-Analyse für mehrere Symbole"""
        results = {}

        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol, period=period)
            except Exception as e:
                results[symbol] = {'error': str(e)}

        return results


if __name__ == "__main__":
    config = {
        'fp16': True,
        'max_new_tokens': 512,
        'temperature': 0.6
    }

    agent = FundamentalAgent(
        model_path="models/fundamental_agent_v1",
        config=config
    )

    result = agent.analyze("AAPL", period="Q")
    print(json.dumps(result, indent=2))
