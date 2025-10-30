import { useState } from 'react';
import { AlertCircle, TrendingUp, Info } from 'lucide-react';
import { StockInput } from './components/StockInput';
import { StockChart } from './components/StockChart';
import { StockStats } from './components/StockStats';
import { ModelComparison } from './components/ModelComparison';
import { ExportData } from './components/ExportData';
import { fetchStockData, predictStockPrice } from './services/stockApi';
import { StockDataResponse, PredictionResponse } from './types/stock';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stockData, setStockData] = useState<StockDataResponse | null>(null);
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null);

  const handleSearch = async (ticker: string, days: number, predictionDays: number) => {
    setIsLoading(true);
    setError(null);
    setStockData(null);
    setPredictions(null);

    try {
      const data = await fetchStockData(ticker, days);
      setStockData(data);

      const predictionData = await predictStockPrice(ticker, predictionDays);
      setPredictions(predictionData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-50">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="h-12 w-12 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg">
              <TrendingUp className="h-7 w-7 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900">
              Stock Price Predictor
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Analyze historical stock data and predict future prices using Linear Regression and LSTM Neural Networks
          </p>
        </header>

        <div className="max-w-7xl mx-auto space-y-8">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <StockInput onSearch={handleSearch} isLoading={isLoading} />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6 flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-red-900 font-semibold mb-1">Error</h3>
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          )}

          {stockData && (
            <>
              <StockStats
                ticker={stockData.ticker}
                currentPrice={stockData.current_price}
                historicalData={stockData.historical_data}
              />

              {predictions && predictions.models && predictions.models.length > 0 && (
                <>
                  <div className="bg-white rounded-xl shadow-lg p-8">
                    <h2 className="text-2xl font-bold text-gray-900 mb-6">Model Predictions Comparison</h2>
                    <ModelComparison
                      models={predictions.models}
                      currentPrice={stockData.current_price}
                    />
                  </div>

                  {predictions.lstm_error && (
                    <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 flex items-start gap-3">
                      <Info className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <h3 className="text-amber-900 font-semibold mb-1">LSTM Model Notice</h3>
                        <p className="text-amber-800 text-sm">{predictions.lstm_error}</p>
                      </div>
                    </div>
                  )}
                </>
              )}

              <div className="bg-white rounded-xl shadow-lg p-8">
                <StockChart
                  ticker={stockData.ticker}
                  historicalData={stockData.historical_data}
                  models={predictions?.models}
                />
              </div>

              {predictions && (
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="text-blue-900 font-semibold mb-1">Disclaimer</h3>
                      <p className="text-blue-800 text-sm">{predictions.disclaimer}</p>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex justify-center">
                <ExportData
                  ticker={stockData.ticker}
                  historicalData={stockData.historical_data}
                  models={predictions?.models}
                />
              </div>
            </>
          )}

          {!stockData && !error && !isLoading && (
            <div className="text-center py-16">
              <div className="h-24 w-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="h-12 w-12 text-gray-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                Ready to Analyze
              </h3>
              <p className="text-gray-500">
                Enter a stock symbol above to view historical data and AI-powered predictions
              </p>
            </div>
          )}
        </div>

        <footer className="mt-16 text-center text-sm text-gray-500">
          <p>
            Data sourced from Yahoo Finance. Predictions are for educational purposes only.
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
