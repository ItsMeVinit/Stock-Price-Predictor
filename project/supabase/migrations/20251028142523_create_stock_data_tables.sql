/*
  # Stock Prediction App Database Schema

  1. New Tables
    - `stock_data`
      - `id` (uuid, primary key)
      - `ticker` (text, stock symbol)
      - `date` (date, trading date)
      - `open` (numeric, opening price)
      - `high` (numeric, highest price)
      - `low` (numeric, lowest price)
      - `close` (numeric, closing price)
      - `volume` (bigint, trading volume)
      - `created_at` (timestamptz, record creation time)
      - Unique constraint on ticker + date combination
    
    - `predictions`
      - `id` (uuid, primary key)
      - `ticker` (text, stock symbol)
      - `prediction_date` (date, date of prediction creation)
      - `target_date` (date, date being predicted)
      - `predicted_price` (numeric, predicted closing price)
      - `confidence_lower` (numeric, lower bound of prediction)
      - `confidence_upper` (numeric, upper bound of prediction)
      - `model_version` (text, version of prediction model used)
      - `created_at` (timestamptz, record creation time)
    
  2. Security
    - Enable RLS on both tables
    - Allow public read access for stock data (public market information)
    - Allow public read access for predictions (public market predictions)
    
  3. Indexes
    - Index on ticker for fast lookups
    - Index on date ranges for historical queries
*/

-- Create stock_data table
CREATE TABLE IF NOT EXISTS stock_data (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ticker text NOT NULL,
  date date NOT NULL,
  open numeric(10, 2) NOT NULL,
  high numeric(10, 2) NOT NULL,
  low numeric(10, 2) NOT NULL,
  close numeric(10, 2) NOT NULL,
  volume bigint NOT NULL,
  created_at timestamptz DEFAULT now(),
  UNIQUE(ticker, date)
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ticker text NOT NULL,
  prediction_date date NOT NULL,
  target_date date NOT NULL,
  predicted_price numeric(10, 2) NOT NULL,
  confidence_lower numeric(10, 2),
  confidence_upper numeric(10, 2),
  model_version text DEFAULT 'v1',
  created_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE stock_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- RLS Policies for stock_data (public read access)
CREATE POLICY "Allow public read access to stock data"
  ON stock_data
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow authenticated read access to stock data"
  ON stock_data
  FOR SELECT
  TO authenticated
  USING (true);

-- RLS Policies for predictions (public read access)
CREATE POLICY "Allow public read access to predictions"
  ON predictions
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow authenticated read access to predictions"
  ON predictions
  FOR SELECT
  TO authenticated
  USING (true);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_data_ticker ON stock_data(ticker);
CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_data_ticker_date ON stock_data(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON predictions(target_date DESC);
