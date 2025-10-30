import { Brain, TrendingUp } from 'lucide-react';
import { ModelPredictions } from '../types/stock';

interface ModelComparisonProps {
  models: ModelPredictions[];
  currentPrice: number;
}

export function ModelComparison({ models, currentPrice }: ModelComparisonProps) {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const getModelIcon = (modelName: string) => {
    if (modelName === 'lstm') {
      return <Brain className="h-5 w-5" />;
    }
    return <TrendingUp className="h-5 w-5" />;
  };

  const getModelLabel = (modelName: string) => {
    const labels: Record<string, string> = {
      'linear_regression': 'Linear Regression',
      'lstm': 'LSTM Neural Network',
    };
    return labels[modelName] || modelName;
  };

  const getModelDescription = (modelName: string) => {
    const descriptions: Record<string, string> = {
      'linear_regression': 'Simple trend-based prediction using historical price movement',
      'lstm': 'Advanced time-series prediction using recurrent neural networks',
    };
    return descriptions[modelName] || 'Predictive model';
  };

  const getModelColor = (modelName: string) => {
    const colors: Record<string, string> = {
      'linear_regression': 'green',
      'lstm': 'amber',
    };
    return colors[modelName] || 'gray';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {models.map((model) => {
        const lastPrediction = model.predictions[model.predictions.length - 1];
        if (!lastPrediction) return null;

        const priceChange = lastPrediction.predicted_price - currentPrice;
        const priceChangePercent = (priceChange / currentPrice) * 100;
        const color = getModelColor(model.model);
        const isPositive = priceChange >= 0;

        return (
          <div
            key={model.model}
            className="bg-white rounded-lg shadow-sm border-2 border-gray-200 p-6"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div
                  className={`h-10 w-10 bg-${color}-100 rounded-lg flex items-center justify-center`}
                >
                  <div className={`text-${color}-600`}>
                    {getModelIcon(model.model)}
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">
                    {getModelLabel(model.model)}
                  </h3>
                  <p className="text-xs text-gray-600 mt-0.5">
                    {getModelDescription(model.model)}
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <p className="text-sm text-gray-600 mb-1">Predicted Price</p>
                <p className="text-2xl font-bold text-gray-900">
                  {formatCurrency(lastPrediction.predicted_price)}
                </p>
              </div>

              <div className="flex items-center gap-2">
                <div
                  className={`flex items-center gap-1 px-2 py-1 rounded ${
                    isPositive ? 'bg-green-100' : 'bg-red-100'
                  }`}
                >
                  <span
                    className={`text-sm font-medium ${
                      isPositive ? 'text-green-700' : 'text-red-700'
                    }`}
                  >
                    {formatCurrency(priceChange)}
                  </span>
                  <span
                    className={`text-sm font-medium ${
                      isPositive ? 'text-green-700' : 'text-red-700'
                    }`}
                  >
                    ({formatPercent(priceChangePercent)})
                  </span>
                </div>
              </div>

              <div className="pt-3 border-t border-gray-200">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">Confidence Range</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-700">
                    {formatCurrency(lastPrediction.confidence_lower)}
                  </span>
                  <span className="text-gray-400">-</span>
                  <span className="text-gray-700">
                    {formatCurrency(lastPrediction.confidence_upper)}
                  </span>
                </div>
              </div>

              <div className="pt-2">
                <div className="text-xs text-gray-500">
                  Forecast: {model.predictions.length} days ahead
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
