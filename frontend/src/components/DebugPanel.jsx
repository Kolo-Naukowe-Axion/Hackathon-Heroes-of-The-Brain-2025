import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Model 2.1 classes: ['Calm', 'Relaxed', 'Negative/Angry', 'Positive/Happy']
const MODEL_CLASSES = ['Calm', 'Relaxed', 'Negative/Angry', 'Positive/Happy'];
const CLASS_COLORS = {
  'Calm': '#3B82F6',           // Blue
  'Relaxed': '#8B5CF6',        // Purple
  'Negative/Angry': '#EF4444', // Red
  'Positive/Happy': '#10B981'  // Green
};

export function DebugPanel({ probabilities, isMock }) {
  const canvasRef = useRef(null);
  const lastProbsRef = useRef(null);
  const [history, setHistory] = useState([]);
  const [lastUpdateTime, setLastUpdateTime] = useState(0);
  const [updateCount, setUpdateCount] = useState(0);
  const maxHistoryLength = 300; // Keep last 300 points (30 seconds at 10Hz) - increased for better context

  // Add current probabilities to history
  useEffect(() => {
    if (probabilities && probabilities.length === 4) {
      const now = Date.now();
      const probsStr = probabilities.map(p => p.toFixed(3)).join(', ');
      
      // Log when probabilities actually change
      if (lastProbsRef.current) {
        const lastProbsStr = lastProbsRef.current.map(p => p.toFixed(3)).join(', ');
        if (probsStr !== lastProbsStr) {
          console.log('[DEBUG PANEL] Probabilities changed!', {
            old: lastProbsStr,
            new: probsStr,
            oldArray: [...lastProbsRef.current],
            newArray: [...probabilities]
          });
        }
      } else {
        console.log('[DEBUG PANEL] First probabilities received:', {
          probs: probsStr,
          array: [...probabilities]
        });
      }
      
      // Update ref
      lastProbsRef.current = [...probabilities];
      
      setLastUpdateTime(now);
      setUpdateCount(prev => prev + 1);
      setHistory(prev => {
        const newHistory = [...prev, { time: now, probs: [...probabilities] }];
        return newHistory.slice(-maxHistoryLength);
      });
    }
  }, [probabilities]);

  // Draw graph
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = (height / 10) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw probability lines
    MODEL_CLASSES.forEach((className, classIdx) => {
      if (history.length < 2) return;

      ctx.strokeStyle = CLASS_COLORS[className];
      ctx.lineWidth = 2;
      ctx.beginPath();

      history.forEach((point, idx) => {
        const x = history.length > 1 ? (width / (history.length - 1)) * idx : width / 2;
        const y = height - (point.probs[classIdx] * height); // Invert Y axis

        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();
    });

    // Draw current values as bars
    if (probabilities && probabilities.length === 4) {
      const barWidth = width / 4;
      probabilities.forEach((prob, idx) => {
        const barHeight = prob * height;
        ctx.fillStyle = CLASS_COLORS[MODEL_CLASSES[idx]] + '80'; // 50% opacity
        ctx.fillRect(idx * barWidth, height - barHeight, barWidth - 2, barHeight);
      });
    }

    // Draw labels on the right
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '12px monospace';
    ctx.textAlign = 'right';
    MODEL_CLASSES.forEach((className, idx) => {
      const y = (height / 4) * (idx + 0.5);
      ctx.fillStyle = CLASS_COLORS[className];
      ctx.fillText(`${className}:`, width - 100, y);
      if (probabilities && probabilities[idx] !== undefined) {
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(`${(probabilities[idx] * 100).toFixed(1)}%`, width - 10, y);
      }
    });

    // Draw time axis labels
    ctx.fillStyle = '#666666';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    if (history.length > 1) {
      const totalTime = history[history.length - 1].time - history[0].time;
      for (let i = 0; i <= 5; i++) {
        const x = (width / 5) * i;
        const timeOffset = (totalTime / 5) * i;
        const secondsAgo = ((totalTime - timeOffset) / 1000).toFixed(1);
        ctx.fillText(`-${secondsAgo}s`, x, height - 5);
      }
    }
  }, [history, probabilities]);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 20 }}
        className="fixed top-20 right-4 w-96 bg-black/95 backdrop-blur-xl border border-white/20 rounded-lg p-4 z-50 shadow-2xl"
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-white font-bold text-lg">DEBUG MODE</h3>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isMock ? 'bg-yellow-500' : 'bg-green-500'}`} />
            <span className="text-xs text-gray-400">
              {isMock ? 'SIMULATION' : 'LIVE'}
            </span>
          </div>
        </div>

        {/* Probability Graph */}
        <div className="mb-4">
          <canvas
            ref={canvasRef}
            width={368}
            height={200}
            className="w-full border border-white/10 rounded"
          />
        </div>

        {/* Current Probabilities Table */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-300 mb-2">Current Probabilities:</h4>
          {probabilities && probabilities.length === 4 ? (
            <div className="space-y-1">
              {MODEL_CLASSES.map((className, idx) => {
                const prob = probabilities[idx];
                const percentage = (prob * 100).toFixed(1);
                return (
                  <div key={className} className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded"
                      style={{ backgroundColor: CLASS_COLORS[className] }}
                    />
                    <span className="text-xs text-gray-400 w-16">{className}:</span>
                    <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
                      <div
                        className="h-full transition-all duration-100"
                        style={{
                          width: `${percentage}%`,
                          backgroundColor: CLASS_COLORS[className]
                        }}
                      />
                    </div>
                    <span className="text-xs text-white w-12 text-right">{percentage}%</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-gray-500">Waiting for data...</p>
          )}
        </div>

        {/* Stats */}
        <div className="mt-4 pt-4 border-t border-white/10">
          <div className="flex justify-between text-xs text-gray-400">
            <span>History: {history.length} pts</span>
            <span>Updates: {updateCount}</span>
          </div>
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>Last update: {lastUpdateTime > 0 ? `${((Date.now() - lastUpdateTime) / 1000).toFixed(1)}s ago` : 'Never'}</span>
            <span className={updateCount > 0 ? 'text-green-400' : 'text-red-400'}>
              {updateCount > 0 ? '● LIVE' : '○ NO DATA'}
            </span>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

