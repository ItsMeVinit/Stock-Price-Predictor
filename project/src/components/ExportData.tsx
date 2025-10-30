import { Download } from 'lucide-react';
import { StockDataPoint, ModelPredictions } from '../types/stock';

interface ExportDataProps {
  ticker: string;
  historicalData: StockDataPoint[];
  models?: ModelPredictions[];
}

export function ExportData({ ticker, historicalData, models }: ExportDataProps) {
  const exportToCSV = () => {
    const headers = ['Date', 'Type', 'Model', 'Price', 'Open', 'High', 'Low', 'Volume', 'Confidence Lower', 'Confidence Upper'];
    const rows = [];

    historicalData.forEach(data => {
      rows.push([
        data.date,
        'Historical',
        '',
        data.close.toFixed(2),
        data.open.toFixed(2),
        data.high.toFixed(2),
        data.low.toFixed(2),
        data.volume,
        '',
        '',
      ]);
    });

    if (models) {
      models.forEach(model => {
        model.predictions.forEach(pred => {
          rows.push([
            pred.date,
            'Predicted',
            model.model,
            pred.predicted_price.toFixed(2),
            '',
            '',
            '',
            '',
            pred.confidence_lower.toFixed(2),
            pred.confidence_upper.toFixed(2),
          ]);
        });
      });
    }

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `${ticker}_stock_data_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToJSON = () => {
    const data = {
      ticker,
      exported_at: new Date().toISOString(),
      historical_data: historicalData,
      models: models || [],
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `${ticker}_stock_data_${new Date().toISOString().split('T')[0]}.json`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex gap-3">
      <button
        onClick={exportToCSV}
        className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium text-gray-700"
      >
        <Download className="h-4 w-4" />
        Export CSV
      </button>
      <button
        onClick={exportToJSON}
        className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium text-gray-700"
      >
        <Download className="h-4 w-4" />
        Export JSON
      </button>
    </div>
  );
}
