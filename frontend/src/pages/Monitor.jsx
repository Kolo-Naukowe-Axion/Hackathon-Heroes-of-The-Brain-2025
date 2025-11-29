import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export default function Monitor() {
    const [data, setData] = useState(null);
    const [history, setHistory] = useState([]);

    useEffect(() => {
        const ws = new WebSocket("ws://localhost:8000/ws");

        ws.onopen = () => {
            console.log("Monitor: Connected to WebSocket");
        };

        ws.onmessage = (event) => {
            try {
                const parsed = JSON.parse(event.data);
                setData(parsed);
                setHistory(prev => [...prev.slice(-50), parsed]); // Keep last 50
            } catch (e) {
                console.error("Monitor: Error parsing data", e);
            }
        };

        return () => ws.close();
    }, []);

    if (!data) return <div className="min-h-screen bg-black text-white flex items-center justify-center">Loading...</div>;

    // data structure: { emotion: "...", probabilities: [neg, neu, pos], history: [...] }
    // probabilities order depends on backend. Based on server.py: [Neg, Neu, Pos]

    const probs = data.probabilities || [0, 0, 0];
    const labels = ["Negative", "Neutral", "Positive"];
    const colors = ["#ef4444", "#9ca3af", "#22c55e"]; // Red, Gray, Green

    return (
        <div className="min-h-screen bg-black text-white p-8 font-mono">
            <h1 className="text-3xl font-bold mb-8 text-center border-b border-gray-800 pb-4">
                Live Emotion Monitor
            </h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Current State */}
                <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
                    <h2 className="text-xl mb-4 text-gray-400">Current Prediction</h2>
                    <div className="text-6xl font-bold mb-2" style={{ color: colors[probs.indexOf(Math.max(...probs))] }}>
                        {data.emotion}
                    </div>
                    <div className="flex gap-4 mt-8">
                        {probs.map((p, i) => (
                            <div key={i} className="flex-1 flex flex-col items-center">
                                <div className="w-full bg-gray-800 rounded-full h-48 relative overflow-hidden">
                                    <motion.div
                                        className="absolute bottom-0 left-0 right-0 w-full"
                                        style={{ backgroundColor: colors[i] }}
                                        animate={{ height: `${p * 100}%` }}
                                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                                    />
                                </div>
                                <span className="mt-2 text-sm text-gray-400">{labels[i]}</span>
                                <span className="font-bold">{(p * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* History Graph (Simple Bars) */}
                <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 flex flex-col">
                    <h2 className="text-xl mb-4 text-gray-400">History (Last 50 updates)</h2>
                    <div className="flex-1 flex items-end gap-1 h-64 border-b border-gray-700 pb-2">
                        {history.map((item, idx) => {
                            const maxProb = Math.max(...(item.probabilities || [0, 0, 0]));
                            const maxIdx = (item.probabilities || [0, 0, 0]).indexOf(maxProb);
                            return (
                                <div
                                    key={idx}
                                    className="flex-1 min-w-[4px] rounded-t-sm"
                                    style={{
                                        height: `${maxProb * 100}%`,
                                        backgroundColor: colors[maxIdx],
                                        opacity: 0.8
                                    }}
                                />
                            );
                        })}
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-2">
                        <span>Oldest</span>
                        <span>Newest</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
