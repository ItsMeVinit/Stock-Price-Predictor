import { createClient } from 'npm:@supabase/supabase-js@2.57.4';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Client-Info, Apikey',
};

interface StockDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
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
    const days = parseInt(url.searchParams.get('days') || '365');

    if (!ticker) {
      return new Response(
        JSON.stringify({ error: 'Ticker symbol is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Check if we have recent data in database
    const fromDate = new Date();
    fromDate.setDate(fromDate.getDate() - days);
    
    const { data: existingData, error: dbError } = await supabase
      .from('stock_data')
      .select('*')
      .eq('ticker', ticker)
      .gte('date', fromDate.toISOString().split('T')[0])
      .order('date', { ascending: true });

    if (dbError) {
      console.error('Database error:', dbError);
    }

    // If we have sufficient recent data, return it
    if (existingData && existingData.length > days * 0.8) {
      const latestData = existingData[existingData.length - 1];
      return new Response(
        JSON.stringify({
          ticker,
          current_price: latestData.close,
          historical_data: existingData.map((d: any) => ({
            date: d.date,
            open: parseFloat(d.open),
            high: parseFloat(d.high),
            low: parseFloat(d.low),
            close: parseFloat(d.close),
            volume: parseInt(d.volume),
          })),
          source: 'cache',
        }),
        {
          status: 200,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    // Fetch from Yahoo Finance API
    const toDate = Math.floor(Date.now() / 1000);
    const fromTimestamp = Math.floor(fromDate.getTime() / 1000);
    
    const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?period1=${fromTimestamp}&period2=${toDate}&interval=1d`;
    
    const response = await fetch(yahooUrl);
    
    if (!response.ok) {
      return new Response(
        JSON.stringify({ error: 'Failed to fetch stock data. Please verify the ticker symbol.' }),
        {
          status: 404,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const data = await response.json();
    
    if (!data.chart?.result?.[0]) {
      return new Response(
        JSON.stringify({ error: 'No data found for this ticker symbol' }),
        {
          status: 404,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      );
    }

    const result = data.chart.result[0];
    const timestamps = result.timestamp;
    const quotes = result.indicators.quote[0];
    
    const historicalData: StockDataPoint[] = [];
    const recordsToInsert = [];

    for (let i = 0; i < timestamps.length; i++) {
      const date = new Date(timestamps[i] * 1000).toISOString().split('T')[0];
      const dataPoint = {
        date,
        open: quotes.open[i] || 0,
        high: quotes.high[i] || 0,
        low: quotes.low[i] || 0,
        close: quotes.close[i] || 0,
        volume: quotes.volume[i] || 0,
      };
      
      historicalData.push(dataPoint);
      recordsToInsert.push({
        ticker,
        ...dataPoint,
      });
    }

    // Store in database for future use
    if (recordsToInsert.length > 0) {
      await supabase
        .from('stock_data')
        .upsert(recordsToInsert, { onConflict: 'ticker,date', ignoreDuplicates: true });
    }

    const currentPrice = historicalData[historicalData.length - 1]?.close || 0;

    return new Response(
      JSON.stringify({
        ticker,
        current_price: currentPrice,
        historical_data: historicalData,
        source: 'live',
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