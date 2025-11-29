import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { motion } from 'framer-motion';
import { emotions } from './utils/emotions';
import { BrainHero } from './components/BrainHero';
import { BackgroundParticles } from './components/BackgroundParticles';
import { PlaylistGrid } from './components/PlaylistGrid';

function App() {
  const [emotionIndex, setEmotionIndex] = useState(0);

  useEffect(() => {
    const handleKeyDown = (event) => {
      switch (event.key.toLowerCase()) {
        case 'n':
          setEmotionIndex(0); // Neutral
          break;
        case 'h':
          setEmotionIndex(1); // Happy
          break;
        case 's':
          setEmotionIndex(2); // Sad
          break;
        case 'a':
          setEmotionIndex(3); // Angry
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const currentEmotion = emotions[emotionIndex];

  return (
    <div className="relative min-h-screen bg-black text-white font-sans selection:bg-white/30">
      {/* Fixed Background Canvas */}
      <div className="fixed inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
          <color attach="background" args={['#000000']} />
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <BrainHero color={currentEmotion.color} />
          <BackgroundParticles color={currentEmotion.color} />
          <OrbitControls enableZoom={false} enablePan={false} />
          <Environment preset="city" />
        </Canvas>
      </div>

      {/* UI Overlay */}
      <div className="relative z-10 flex flex-col items-center w-full">
        {/* Hero Section Spacer */}
        <div className="h-screen flex flex-col items-center justify-start pt-32 pointer-events-none">
          <motion.h1
            key={currentEmotion.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.8 }}
            className="text-6xl md:text-8xl font-bold tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white to-white/50"
            style={{ textShadow: `0 0 30px ${currentEmotion.color}` }}
          >
            {currentEmotion.name}
          </motion.h1>
          <p className="mt-4 text-xl text-gray-400">Current Vibe</p>
        </div>

        {/* Playlists Section */}
        <div className="w-full min-h-screen bg-gradient-to-b from-transparent to-black/80 flex flex-col items-center py-20">
          <h2 className="text-4xl font-bold mb-10">Recommended Playlists</h2>
          <PlaylistGrid color={currentEmotion.color} />
        </div>
      </div>
    </div>
  );
}

export default App;
