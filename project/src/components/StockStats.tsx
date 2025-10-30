import { TrendingUp, TrendingDown, DollarSign } from 'lucide-react';
import { StockDataPoint } from '../types/stock';

interface StockStatsProps {
  ticker: string;
  currentPrice: number;
  historicalData: StockDataPoint[];
}

export function StockStats({ ticker, currentPrice, historicalData }: StockStatsProps) {
  const firstPrice = historicalData[0]?.close || currentPrice;
  const priceChange = currentPrice - firstPrice;
  const priceChangePercent = (priceChange / firstPrice) * 100;

  const highPrice = Math.max(...historicalData.map(d => d.high));
  const lowPrice = Math.min(...historicalData.map(d => d.low));
  const avgVolume = Math.round(
    historicalData.reduce((sum, d) => sum + d.volume, 0) / historicalData.length
  );

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatVolume = (value: number) => {
    if (value >= 1000000000) {
      return `${(value / 1000000000).toFixed(2)}B`;
    } else if (value >= 1000000) {
      return `${(value / 1000000).toFixed(2)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(2)}K`;
    }
    return value.toString();
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">Current Price</p>
            <p className="text-2xl font-bold text-gray-900 mt-1">
              {formatCurrency(currentPrice)}
            </p>
            <div className="flex items-center mt-2">
              {priceChange >= 0 ? (
                <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-600 mr-1" />
              )}
              <span
                className={`text-sm font-medium ${
                  priceChange >= 0 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {priceChange >= 0 ? '+' : ''}
                {formatCurrency(priceChange)} ({priceChangePercent.toFixed(2)}%)
              </span>
            </div>
          </div>
          <div className="h-12 w-12 bg-blue-100 rounded-full flex items-center justify-center">
            <DollarSign className="h-6 w-6 text-blue-600" />
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div>
          <p className="text-sm font-medium text-gray-600">Period High</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {formatCurrency(highPrice)}
          </p>
          <p className="text-sm text-gray-500 mt-2">Highest price in range</p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div>
          <p className="text-sm font-medium text-gray-600">Period Low</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {formatCurrency(lowPrice)}
          </p>
          <p className="text-sm text-gray-500 mt-2">Lowest price in range</p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div>
          <p className="text-sm font-medium text-gray-600">Average Volume</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {formatVolume(avgVolume)}
          </p>
          <p className="text-sm text-gray-500 mt-2">Daily trading volume</p>
        </div>
      </div>
    </div>
  );
}
