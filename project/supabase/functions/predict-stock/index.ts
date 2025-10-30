import { createClient } from 'npm:@supabase/supabase-js@2.57.4';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Client-Info, Apikey',
};

interface Prediction {
  date: string;
  predicted_price: number;
  confidence_lower: number;
  confidence_upper: number;
}

interface ModelPredictions {
  model: string;
  predictions: Prediction[];
}

function normalizeData(data: number[]): { normalized: number[], min: number, max: number } {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;
  const normalized = data.map(val => (val - min) / range);
  return { normalized, min, max };
}

function denormalize(normalizedVal: number, min: number, max: number): number {
  return normalizedVal * (max - min) + min;
}

function predictLinearRegression(historicalPrices: number[], daysToPredict: number): Prediction[] {
  const predictions: Prediction[] = [];
  const n = historicalPrices.length;

  if (n < 10) {
    throw new Error('Insufficient historical data for prediction');
  }

  const windowSize = Math.min(30, Math.floor(n / 3));
  const recentPrices = historicalPrices.slice(-windowSize);

  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

  for (let i = 0; i < recentPrices.length; i++) {
    sumX += i;
    sumY += recentPrices[i];
    sumXY += i * recentPrices[i];
    sumX2 += i * i;
  }

  const slope = (recentPrices.length * sumXY - sumX * sumY) /
                (recentPrices.length * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / recentPrices.length;

  const mean = sumY / recentPrices.length;
  const variance = recentPrices.reduce((acc, price) => acc + Math.pow(price - mean, 2), 0) / recentPrices.length;
  const stdDev = Math.sqrt(variance);

  const lastPrice = historicalPrices[n - 1];
  const today = new Date();

  for (let i = 1; i <= daysToPredict; i++) {
    const predictedPrice = slope * (recentPrices.length + i) + intercept;

    const maxChange = lastPrice * 0.15;
    const clampedPrice = Math.max(
      lastPrice - maxChange,
      Math.min(lastPrice + maxChange, predictedPrice)
    );

    const confidenceMultiplier = 1 + (i / daysToPredict) * 0.5;
    const confidenceRange = stdDev * confidenceMultiplier * 1.96;

    const predictionDate = new Date(today);
    predictionDate.setDate(today.getDate() + i);

    predictions.push({
      date: predictionDate.toISOString().split('T')[0],
      predicted_price: Math.max(0, clampedPrice),
      confidence_lower: Math.max(0, clampedPrice - confidenceRange),
      confidence_upper: clampedPrice + confidenceRange,
    });
  }

  return predictions;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function tanh(x: number): number {
  return Math.tanh(x);
}

class SimpleLSTMCell {
  private hiddenSize: number;
  private Wf: number[][];
  private Wi: number[][];
  private Wc: number[][];
  private Wo: number[][];

  constructor(inputSize: number, hiddenSize: number) {
    this.hiddenSize = hiddenSize;
    this.Wf = this.initWeights(hiddenSize, inputSize + hiddenSize);
    this.Wi = this.initWeights(hiddenSize, inputSize + hiddenSize);
    this.Wc = this.initWeights(hiddenSize, inputSize + hiddenSize);
    this.Wo = this.initWeights(hiddenSize, inputSize + hiddenSize);
  }

  private initWeights(rows: number, cols: number): number[][] {
    const weights: number[][] = [];
    const scale = Math.sqrt(2.0 / (rows + cols));
    for (let i = 0; i < rows; i++) {
      weights[i] = [];
      for (let j = 0; j < cols; j++) {
        weights[i][j] = (Math.random() - 0.5) * 2 * scale;
      }
    }
    return weights;
  }

  private matMul(weights: number[][], input: number[]): number[] {
    const result: number[] = [];
    for (let i = 0; i < weights.length; i++) {
      let sum = 0;
      for (let j = 0; j < input.length; j++) {
        sum += weights[i][j] * input[j];
      }
      result[i] = sum;
    }
    return result;
  }

  forward(x: number, h: number[], c: number[]): { h: number[], c: number[] } {
    const combined = [x, ...h];

    const ft = this.matMul(this.Wf, combined).map(sigmoid);
    const it = this.matMul(this.Wi, combined).map(sigmoid);
    const cTilde = this.matMul(this.Wc, combined).map(tanh);
    const ot = this.matMul(this.Wo, combined).map(sigmoid);

    const newC = c.map((cVal, i) => ft[i] * cVal + it[i] * cTilde[i]);
    const newH = newC.map((cVal, i) => ot[i] * tanh(cVal));

    return { h: newH, c: newC };
  }
}

class SimpleLSTM {
  private cell: SimpleLSTMCell;
  private hiddenSize: number;
  private outputWeights: number[];

  constructor(hiddenSize: number = 50) {
    this.hiddenSize = hiddenSize;
    this.cell = new SimpleLSTMCell(1, hiddenSize);
    this.outputWeights = Array(hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 0.1);
  }

  train(data: number[], epochs: number = 100) {
    const learningRate = 0.001;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let h = Array(this.hiddenSize).fill(0);
      let c = Array(this.hiddenSize).fill(0);

      for (let i = 0; i < data.length - 1; i++) {
        const result = this.cell.forward(data[i], h, c);
        h = result.h;
        c = result.c;

        const predicted = h.reduce((sum, val, idx) => sum + val * this.outputWeights[idx], 0);
        const error = data[i + 1] - predicted;

        for (let j = 0; j < this.hiddenSize; j++) {
          this.outputWeights[j] += learningRate * error * h[j];
        }
      }
    }
  }

  predict(lastValues: number[], steps: number): number[] {
    const predictions: number[] = [];
    let h = Array(this.hiddenSize).fill(0);
    let c = Array(this.hiddenSize).fill(0);

    for (const val of lastValues) {
      const result = this.cell.forward(val, h, c);
      h = result.h;
      c = result.c;
    }

    let lastInput = lastValues[lastValues.length - 1];

    for (let i = 0; i < steps; i++) {
      const result = this.cell.forward(lastInput, h, c);
      h = result.h;
      c = result.c;

      const predicted = h.reduce((sum, val, idx) => sum + val * this.outputWeights[idx], 0);
      predictions.push(predicted);
      lastInput = predicted;
    }

    return predictions;
  }
}

function predictLSTM(historicalPrices: number[], daysToPredict: number): Prediction[] {
  const predictions: Prediction[] = [];
  const n = historicalPrices.length;

  if (n < 60) {
    throw new Error('LSTM requires at least 60 days of historical data');
  }

  const { normalized, min, max } = normalizeData(historicalPrices);

  const lstm = new SimpleLSTM(50);
  lstm.train(normalized, 50);

  const lookbackWindow = Math.min(60, n);
  const lastValues = normalized.slice(-lookbackWindow);
  const normalizedPredictions = lstm.predict(lastValues, daysToPredict);

  const actualPredictions = normalizedPredictions.map(val => denormalize(val, min, max));

  const recentPrices = historicalPrices.slice(-30);
  const mean = recentPrices.reduce((sum, price) => sum + price, 0) / recentPrices.length;
  const variance = recentPrices.reduce((acc, price) => acc + Math.pow(price - mean, 2), 0) / recentPrices.length;
  const stdDev = Math.sqrt(variance);

  const today = new Date();
  const lastPrice = historicalPrices[n - 1];

  for (let i = 0; i < daysToPredict; i++) {
    const rawPrediction = actualPredictions[i];

    const smoothingFactor = 0.7 + (0.3 * Math.exp(-i / 10));
    const smoothedPrediction = rawPrediction * smoothingFactor + lastPrice * (1 - smoothingFactor);

    const maxDeviation = lastPrice * 0.20;
    const clampedPrice = Math.max(
      lastPrice - maxDeviation,
      Math.min(lastPrice + maxDeviation, smoothedPrediction)
    );

    const confidenceMultiplier = 1 + (i / daysToPredict) * 0.6;
    const confidenceRange = stdDev * confidenceMultiplier * 2.0;

    const predictionDate = new Date(today);
    predictionDate.setDate(today.getDate() + i + 1);

    predictions.push({
      date: predictionDate.toISOString().split('T')[0],
      predicted_price: Math.max(0, clampedPrice),
      confidence_lower: Math.max(0, clampedPrice - confidenceRange),
      confidence_upper: clampedPrice + confidenceRange,
    });
  }

  return predictions;
}

Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const url = new URL(req.url);
    const ticker = url.searchParams.get('ticker')?.toUpperCase();
    const days = parseInt(url.searchParams.get('days') || '30');

    if (!ticker) {
      return new Response(
        JSON.stringify({ error: 'Ticker symbol is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    if (days < 1 || days > 90) {
      return new Response(
        JSON.stringify({ error: 'Days must be between 1 and 90' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    const { data: historicalData, error: dbError } = await supabase
      .from('stock_data')
      .select('close, date')
      .eq('ticker', ticker)
      .order('date', { ascending: true })
      .limit(365);

    if (dbError) {
      console.error('Database error:', dbError);
      return new Response(
        JSON.stringify({ error: 'Failed to fetch historical data' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    if (!historicalData || historicalData.length < 10) {
      return new Response(
        JSON.stringify({
          error: 'Insufficient historical data. Please fetch stock data first.'
        }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const closingPrices = historicalData.map((d: any) => parseFloat(d.close));

    const lrPredictions = predictLinearRegression(closingPrices, days);

    let lstmPredictions: Prediction[] = [];
    let lstmError: string | null = null;

    try {
      if (closingPrices.length >= 60) {
        lstmPredictions = predictLSTM(closingPrices, days);
      } else {
        lstmError = 'Insufficient data for LSTM (requires 60+ days)';
      }
    } catch (error) {
      lstmError = error.message;
      console.error('LSTM prediction error:', error);
    }

    const predictionDate = new Date().toISOString().split('T')[0];
    const recordsToInsert = [];

    lrPredictions.forEach(pred => {
      recordsToInsert.push({
        ticker,
        prediction_date: predictionDate,
        target_date: pred.date,
        predicted_price: pred.predicted_price,
        confidence_lower: pred.confidence_lower,
        confidence_upper: pred.confidence_upper,
        model_version: 'linear_regression',
      });
    });

    if (lstmPredictions.length > 0) {
      lstmPredictions.forEach(pred => {
        recordsToInsert.push({
          ticker,
          prediction_date: predictionDate,
          target_date: pred.date,
          predicted_price: pred.predicted_price,
          confidence_lower: pred.confidence_lower,
          confidence_upper: pred.confidence_upper,
          model_version: 'lstm',
        });
      });
    }

    if (recordsToInsert.length > 0) {
      await supabase.from('predictions').insert(recordsToInsert);
    }

    const models: ModelPredictions[] = [
      {
        model: 'linear_regression',
        predictions: lrPredictions,
      }
    ];

    if (lstmPredictions.length > 0) {
      models.push({
        model: 'lstm',
        predictions: lstmPredictions,
      });
    }

    return new Response(
      JSON.stringify({
        ticker,
        prediction_days: days,
        models,
        lstm_error: lstmError,
        disclaimer: 'These predictions are for educational purposes only and should not be used for actual trading decisions.',
      }),
      {
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('Error:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error', details: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
