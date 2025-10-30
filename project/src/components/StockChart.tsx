import { useMemo } from 'react';
import { StockDataPoint, Prediction, ModelPredictions } from '../types/stock';

interface StockChartProps {
  historicalData: StockDataPoint[];
  models?: ModelPredictions[];
  ticker: string;
}

export function StockChart({ historicalData, models, ticker }: StockChartProps) {
  const { minPrice, maxPrice, priceRange } = useMemo(() => {
    const allPrices = [...historicalData.map(d => d.close)];

    models?.forEach(model => {
      allPrices.push(
        ...model.predictions.map(p => p.predicted_price),
        ...model.predictions.map(p => p.confidence_upper),
        ...model.predictions.map(p => p.confidence_lower)
      );
    });

    const min = Math.min(...allPrices);
    const max = Math.max(...allPrices);
    const range = max - min;
    const padding = range * 0.1;

    return {
      minPrice: min - padding,
      maxPrice: max + padding,
      priceRange: range + (padding * 2),
    };
  }, [historicalData, models]);

  const getYPosition = (price: number) => {
    return ((maxPrice - price) / priceRange) * 100;
  };

  const hasPredictions = models && models.length > 0 && models.some(m => m.predictions.length > 0);
  const historicalWidth = hasPredictions ? 70 : 100;
  const predictionWidth = hasPredictions ? 30 : 0;

  const getXPosition = (index: number, total: number, isPrediction: boolean) => {
    if (isPrediction) {
      return historicalWidth + (index / total) * predictionWidth;
    }
    return (index / total) * historicalWidth;
  };

  const historicalPath = historicalData
    .map((point, index) => {
      const x = getXPosition(index, historicalData.length - 1, false);
      const y = getYPosition(point.close);
      return `${index === 0 ? 'M' : 'L'} ${x},${y}`;
    })
    .join(' ');

  const modelColors: Record<string, { line: string, fill: string, label: string }> = {
    'linear_regression': { line: '#22c55e', fill: '#22c55e', label: 'Linear Regression' },
    'lstm': { line: '#f59e0b', fill: '#f59e0b', label: 'LSTM Neural Network' },
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(price);
  };

  const priceLabels = [maxPrice, (maxPrice + minPrice) / 2, minPrice];

  return (
    <div className="w-full h-full">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{ticker}</h3>
          <p className="text-sm text-gray-600">
            Historical {hasPredictions ? 'and Predicted ' : ''}Price Chart
          </p>
        </div>
        <div className="flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-blue-600"></div>
            <span className="text-gray-700">Historical</span>
          </div>
          {models?.map((model) => (
            <div key={model.model} className="flex items-center gap-2">
              <div
                className="w-4 h-0.5 border-t-2 border-dashed"
                style={{
                  borderColor: modelColors[model.model]?.line || '#6b7280'
                }}
              ></div>
              <span className="text-gray-700">
                {modelColors[model.model]?.label || model.model}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="relative w-full" style={{ height: '400px' }}>
        <div className="absolute left-0 top-0 bottom-0 w-16 flex flex-col justify-between text-xs text-gray-600">
          {priceLabels.map((price, i) => (
            <div key={i} className="text-right pr-2">
              {formatPrice(price)}
            </div>
          ))}
        </div>

        <div className="absolute left-16 right-0 top-0 bottom-0">
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="w-full h-full"
          >
            <defs>
              <linearGradient id="historicalGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.2" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
              </linearGradient>
              {models?.map((model) => (
                <linearGradient
                  key={`gradient-${model.model}`}
                  id={`gradient-${model.model}`}
                  x1="0%"
                  y1="0%"
                  x2="0%"
                  y2="100%"
                >
                  <stop offset="0%" stopColor={modelColors[model.model]?.fill || '#6b7280'} stopOpacity="0.1" />
                  <stop offset="100%" stopColor={modelColors[model.model]?.fill || '#6b7280'} stopOpacity="0" />
                </linearGradient>
              ))}
            </defs>

            {priceLabels.map((_, i) => (
              <line
                key={i}
                x1="0"
                y1={i * 50}
                x2="100"
                y2={i * 50}
                stroke="#e5e7eb"
                strokeWidth="0.2"
              />
            ))}

            {hasPredictions && (
              <line
                x1={historicalWidth}
                y1="0"
                x2={historicalWidth}
                y2="100"
                stroke="#d1d5db"
                strokeWidth="0.3"
                strokeDasharray="1,1"
              />
            )}

            {models?.map((model) => {
              const confidenceArea = model.predictions.map((point, index) => {
                const x = getXPosition(index, model.predictions.length - 1, true);
                const yUpper = getYPosition(point.confidence_upper);
                const yLower = getYPosition(point.confidence_lower);
                return { x, yUpper, yLower };
              });

              const confidencePath = confidenceArea.length
                ? `M ${confidenceArea[0].x},${confidenceArea[0].yUpper} ` +
                  confidenceArea.map(p => `L ${p.x},${p.yUpper}`).join(' ') +
                  ' ' +
                  confidenceArea
                    .slice()
                    .reverse()
                    .map(p => `L ${p.x},${p.yLower}`)
                    .join(' ') +
                  ' Z'
                : '';

              return confidencePath ? (
                <path
                  key={`confidence-${model.model}`}
                  d={confidencePath}
                  fill={modelColors[model.model]?.fill || '#6b7280'}
                  opacity="0.15"
                />
              ) : null;
            })}

            <path
              d={`${historicalPath} L ${historicalWidth},100 L 0,100 Z`}
              fill="url(#historicalGradient)"
            />

            <path
              d={historicalPath}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="0.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {models?.map((model) => {
              const predictionPath = model.predictions
                .map((point, index) => {
                  const x = getXPosition(index, model.predictions.length - 1, true);
                  const y = getYPosition(point.predicted_price);
                  return `${index === 0 ? 'M' : 'L'} ${x},${y}`;
                })
                .join(' ');

              return predictionPath ? (
                <path
                  key={`prediction-${model.model}`}
                  d={predictionPath}
                  fill="none"
                  stroke={modelColors[model.model]?.line || '#6b7280'}
                  strokeWidth="0.5"
                  strokeDasharray="1,1"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              ) : null;
            })}
          </svg>
        </div>
      </div>

      <div className="mt-4 flex justify-between text-xs text-gray-600">
        <span>
          {new Date(historicalData[0]?.date).toLocaleDateString()}
        </span>
        {hasPredictions && (
          <span className="text-gray-700 font-medium">
            Forecast until {new Date(models![0].predictions[models![0].predictions.length - 1]?.date).toLocaleDateString()}
          </span>
        )}
        <span>
          {new Date(historicalData[historicalData.length - 1]?.date).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
}
