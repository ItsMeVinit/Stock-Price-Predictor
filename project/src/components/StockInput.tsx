import { useState } from 'react';
import { Search, TrendingUp } from 'lucide-react';

interface StockInputProps {
  onSearch: (ticker: string, days: number, predictionDays: number) => void;
  isLoading: boolean;
}

export function StockInput({ onSearch, isLoading }: StockInputProps) {
  const [ticker, setTicker] = useState('');
  const [days, setDays] = useState('365');
  const [predictionDays, setPredictionDays] = useState('30');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      onSearch(ticker.trim().toUpperCase(), parseInt(days), parseInt(predictionDays));
    }
  };

  const popularStocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA'];

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="ticker" className="block text-sm font-medium text-gray-700 mb-2">
            Stock Symbol
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <TrendingUp className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              id="ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="Enter ticker symbol (e.g., AAPL)"
              className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
              disabled={isLoading}
              required
            />
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <span className="text-xs text-gray-600">Popular:</span>
            {popularStocks.map((stock) => (
              <button
                key={stock}
                type="button"
                onClick={() => setTicker(stock)}
                className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded transition-colors"
                disabled={isLoading}
              >
                {stock}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label htmlFor="days" className="block text-sm font-medium text-gray-700 mb-2">
              Historical Data (days)
            </label>
            <select
              id="days"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              className="block w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            >
              <option value="30">1 Month</option>
              <option value="90">3 Months</option>
              <option value="180">6 Months</option>
              <option value="365">1 Year</option>
            </select>
          </div>

          <div>
            <label htmlFor="predictionDays" className="block text-sm font-medium text-gray-700 mb-2">
              Prediction Period (days)
            </label>
            <select
              id="predictionDays"
              value={predictionDays}
              onChange={(e) => setPredictionDays(e.target.value)}
              className="block w-full px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            >
              <option value="7">1 Week</option>
              <option value="14">2 Weeks</option>
              <option value="30">1 Month</option>
              <option value="60">2 Months</option>
              <option value="90">3 Months</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading || !ticker.trim()}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {isLoading ? (
            <>
              <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Search className="h-5 w-5" />
              <span>Analyze & Predict</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
