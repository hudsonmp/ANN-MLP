import { useEffect, useRef, useState } from 'react';
import getStroke from 'perfect-freehand';

interface Point {
  x: number;
  y: number;
  pressure?: number;
}

interface DrawingCanvasProps {
  onPredict: (imageData: string) => void;
}

export default function DrawingCanvas({ onPredict }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [points, setPoints] = useState<Point[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas to white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Enable touch events
    canvas.style.touchAction = 'none';
  }, []);

  const draw = (ctx: CanvasRenderingContext2D, points: Point[]) => {
    const stroke = getStroke(points, {
      size: 16,
      thinning: 0.5,
      smoothing: 0.5,
      streamline: 0.5,
    });

    ctx.fillStyle = 'black';
    ctx.beginPath();
    for (const [x, y] of stroke) {
      ctx.lineTo(x, y);
    }
    ctx.fill();
  };

  const addPoint = (x: number, y: number, pressure = 0.5) => {
    setPoints((prevPoints) => {
      const newPoints = [...prevPoints, { x, y, pressure }];
      
      const canvas = canvasRef.current;
      if (!canvas) return newPoints;

      const ctx = canvas.getContext('2d');
      if (!ctx) return newPoints;

      // Clear canvas and redraw
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      draw(ctx, newPoints);
      
      return newPoints;
    });
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    let x, y;

    if ('touches' in e) {
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.clientX - rect.left;
      y = e.clientY - rect.top;
    }

    setIsDrawing(true);
    addPoint(x, y);
  };

  const continueDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    let x, y;

    if ('touches' in e) {
      e.preventDefault(); // Prevent scrolling while drawing
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.clientX - rect.left;
      y = e.clientY - rect.top;
    }

    addPoint(x, y);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPoints([]);
  };

  const handlePredict = async () => {
    if (points.length === 0) return;
    
    setIsPredicting(true);
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      const imageData = canvas.toDataURL('image/png');
      await onPredict(imageData);
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="bg-white rounded-2xl shadow-lg transition-all duration-300 ease-in-out
                     hover:shadow-xl border-2 border-gray-100 touch-none"
          onMouseDown={startDrawing}
          onMouseMove={continueDrawing}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={continueDrawing}
          onTouchEnd={stopDrawing}
        />
        {points.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-400 pointer-events-none">
            Draw a digit here
          </div>
        )}
      </div>
      <div className="flex gap-4">
        <button
          onClick={clearCanvas}
          disabled={points.length === 0 || isPredicting}
          className="px-6 py-2.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-full
                     transition-all duration-300 ease-in-out
                     hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed
                     focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
        >
          Clear
        </button>
        <button
          onClick={handlePredict}
          disabled={points.length === 0 || isPredicting}
          className="px-6 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-full
                     transition-all duration-300 ease-in-out
                     hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2
                     flex items-center gap-2"
        >
          {isPredicting ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Processing...
            </>
          ) : (
            'Predict'
          )}
        </button>
      </div>
    </div>
  );
} 