'use client';

import { useState } from 'react';
import DrawingCanvas from '@/components/DrawingCanvas';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [prediction, setPrediction] = useState<{ digit: number; confidence: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (imageData: string) => {
    try {
      setError(null);
      const response = await fetch(`${API_URL}/api/predict/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.error || 
          `Failed to get prediction (HTTP ${response.status})`
        );
      }

      const result = await response.json();
      if (!result.digit && result.digit !== 0) {
        throw new Error('Invalid prediction result');
      }

      setPrediction(result);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(
        err instanceof Error 
          ? err.message 
          : 'Failed to connect to the prediction service. Please try again.'
      );
      setPrediction(null);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-4xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl md:text-6xl mb-4">
            Digit Recognition
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Draw any digit from 0 to 9, and watch as our AI instantly recognizes it.
            Experience the power of machine learning in real-time.
          </p>
        </div>

        <div className="bg-white rounded-3xl shadow-xl p-8 mb-8 transition-all duration-300 hover:shadow-2xl">
          <DrawingCanvas onPredict={handlePredict} />
        </div>
        
        <div className="space-y-6">
          {prediction && (
            <div className="bg-white rounded-2xl p-8 shadow-lg transition-all duration-300">
              <div className="text-center">
                <span className="text-7xl font-bold text-blue-600 block mb-4 animate-fade-in">
                  {prediction.digit}
                </span>
                <div className="flex items-center justify-center gap-2">
                  <div className="h-2 w-full max-w-md bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-600 rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-gray-600 min-w-[4rem]">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="mt-2 text-gray-600">
                  Confidence Level
                </p>
              </div>
            </div>
          )}
          
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-2xl p-6">
              <div className="flex items-center gap-3">
                <svg className="h-5 w-5 text-red-600" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <p className="text-red-600 font-medium">{error}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
