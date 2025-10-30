import { StockDataResponse, PredictionResponse } from '../types/stock';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

const headers = {
  'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
  'Content-Type': 'application/json',
};

export async function fetchStockData(ticker: string, days: number = 365): Promise<StockDataResponse> {
  const apiUrl = `${SUPABASE_URL}/functions/v1/get-stock-data?ticker=${encodeURIComponent(ticker)}&days=${days}`;

  const response = await fetch(apiUrl, { headers });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to fetch stock data');
  }

  return response.json();
}

export async function predictStockPrice(ticker: string, days: number = 30): Promise<PredictionResponse> {
  const apiUrl = `${SUPABASE_URL}/functions/v1/predict-stock?ticker=${encodeURIComponent(ticker)}&days=${days}`;

  const response = await fetch(apiUrl, { headers });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to generate predictions');
  }

  return response.json();
}
