"""
News Fetcher - API Client for Financial News
"""

import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests


class NewsFetcher:
    """
    News Fetcher für Financial News APIs.

    Unterstützt:
    - Finnhub
    - NewsAPI
    - Serper (Google News)
    """

    def __init__(
        self,
        finnhub_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        serper_key: Optional[str] = None
    ):
        """
        Args:
            finnhub_key: Finnhub API key
            newsapi_key: NewsAPI.org key
            serper_key: Serper.dev key
        """
        self.finnhub_key = finnhub_key or os.getenv('FINNHUB_API_KEY')
        self.newsapi_key = newsapi_key or os.getenv('NEWS_API_KEY')
        self.serper_key = serper_key or os.getenv('SERPER_API_KEY')

    def get_news(
        self,
        symbol: str,
        days: int = 7,
        source: str = "auto"
    ) -> List[Dict]:
        """
        Hole News für ein Symbol

        Args:
            symbol: Stock symbol
            days: Anzahl Tage zurück
            source: "finnhub", "newsapi", "serper", or "auto"

        Returns:
            Liste von News-Artikeln
        """
        # Auto-select source based on available keys
        if source == "auto":
            if self.finnhub_key:
                source = "finnhub"
            elif self.newsapi_key:
                source = "newsapi"
            elif self.serper_key:
                source = "serper"
            else:
                return []

        # Fetch from selected source
        if source == "finnhub":
            return self._fetch_finnhub(symbol, days)
        elif source == "newsapi":
            return self._fetch_newsapi(symbol, days)
        elif source == "serper":
            return self._fetch_serper(symbol, days)
        else:
            return []

    def _fetch_finnhub(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from Finnhub"""
        if not self.finnhub_key:
            return []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': self.finnhub_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json()

            # Standardize format
            standardized = []
            for article in articles:
                standardized.append({
                    'title': article.get('headline', ''),
                    'description': article.get('summary', ''),
                    'content': article.get('summary', ''),
                    'source': article.get('source', 'Finnhub'),
                    'publishedAt': datetime.fromtimestamp(
                        article.get('datetime', 0)
                    ).isoformat(),
                    'url': article.get('url', '')
                })

            return standardized

        except Exception as e:
            print(f"Error fetching from Finnhub: {e}")
            return []

    def _fetch_newsapi(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        if not self.newsapi_key:
            return []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': self.newsapi_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = data.get('articles', [])

            # Already in standard format
            return articles[:50]  # Limit to 50

        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []

    def _fetch_serper(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from Serper (Google News)"""
        if not self.serper_key:
            return []

        url = "https://google.serper.dev/news"
        payload = {
            'q': f'{symbol} stock',
            'num': 50
        }
        headers = {
            'X-API-KEY': self.serper_key,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Standardize format
            standardized = []
            for article in data.get('news', []):
                standardized.append({
                    'title': article.get('title', ''),
                    'description': article.get('snippet', ''),
                    'content': article.get('snippet', ''),
                    'source': article.get('source', 'Google News'),
                    'publishedAt': article.get('date', datetime.now().isoformat()),
                    'url': article.get('link', '')
                })

            return standardized

        except Exception as e:
            print(f"Error fetching from Serper: {e}")
            return []

    def deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []

        unique_articles = []
        seen_titles = set()

        for article in articles:
            title = article.get('title', '').lower().strip()

            # Simple deduplication by title
            if title and title not in seen_titles:
                unique_articles.append(article)
                seen_titles.add(title)

        return unique_articles


if __name__ == "__main__":
    fetcher = NewsFetcher()
    news = fetcher.get_news("AAPL", days=7)
    print(f"Found {len(news)} articles")
    if news:
        print("\nFirst article:")
        print(f"Title: {news[0]['title']}")
        print(f"Source: {news[0]['source']}")
        print(f"Date: {news[0]['publishedAt']}")
