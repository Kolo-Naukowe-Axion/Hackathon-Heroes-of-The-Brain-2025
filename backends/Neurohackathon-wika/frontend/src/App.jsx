import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import SpotifyPlayer from 'react-spotify-web-playback';

import { emotions } from './utils/emotions';
import { BrainHero } from './components/BrainHero';
import { BackgroundParticles } from './components/BackgroundParticles';
import { redirectToAuthCodeFlow, getAccessToken } from './utils/auth';
import { fetchPlaylistTracks } from './utils/spotify';

// --- KONFIGURACJA ---
const CLIENT_ID = "00ea48b1e2144865828b75c3d4746b7c"; // <--- PAMIĘTAJ O WPISANIU ID!
const REDIRECT_URI = "http://127.0.0.1:5173/";
const SCOPES = ["streaming", "user-read-email", "user-read-private", "user-modify-playback-state"];

function App() {
  const [emotionIndex, setEmotionIndex] = useState(0); // 0 = Neutral
  const [token, setToken] = useState("");
  const [play, setPlay] = useState(false);
  const [activeUri, setActiveUri] = useState("");
  const [playlistTracks, setPlaylistTracks] = useState({});
  const [playerError, setPlayerError] = useState(null);

  // Logowanie PKCE
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");

    // Check for existing token
    const storedToken = window.localStorage.getItem("token");
    if (storedToken) {
      setToken(storedToken);
    }

    if (code && !storedToken) {
      getAccessToken(CLIENT_ID, code, REDIRECT_URI).then((accessToken) => {
        if (accessToken) {
          window.localStorage.setItem("token", accessToken);
          setToken(accessToken);
          window.history.replaceState({}, document.title, "/");
        }
      });
    }
  }, []);

  const handleLogin = () => {
    redirectToAuthCodeFlow(CLIENT_ID, REDIRECT_URI, SCOPES);
  };

  const handleLogout = () => {
    setToken("");
    window.localStorage.removeItem("token");
    window.location.href = "/"; // Refresh to clear state cleanly
  };

  // Klawiatura (Symulacja EEG)
  // N=Neutral, C=Calm, H=Happy, S=Sad, A=Angry
  useEffect(() => {
    const handleKeyDown = (event) => {
      switch (event.key.toLowerCase()) {
        case 'n': setEmotionIndex(0); break; // Neutral
        case 'c': setEmotionIndex(1); break; // Calm
        case 'h': setEmotionIndex(2); break; // Happy
        case 's': setEmotionIndex(3); break; // Sad
        case 'a': setEmotionIndex(4); break; // Angry
        default: break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const currentEmotion = emotions[emotionIndex];

  // Kiedy zmienia się emocja, automatycznie wybierz PIERWSZĄ playlistę z listy
  useEffect(() => {
    if (currentEmotion && currentEmotion.playlists.length > 0) {
      setActiveUri(currentEmotion.playlists[0].uri);
      setPlay(true);
    }
  }, [emotionIndex]);

  // Fetch tracks for current emotion's playlists
  useEffect(() => {
    if (token && currentEmotion) {
      currentEmotion.playlists.forEach(playlist => {
        // Only fetch if we haven't already (optional optimization, but good for rate limits)
        if (!playlistTracks[playlist.uri]) {
          fetchPlaylistTracks(token, playlist.uri).then(tracks => {
            if (tracks && tracks.length > 0) {
              setPlaylistTracks(prev => ({
                ...prev,
                [playlist.uri]: tracks
              }));
            }
          });
        }
      });
    }
  }, [currentEmotion, token]);

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-white/30">

      {/* 1. SEKCJA 3D (GÓRA) */}
      <div className="relative h-[70vh] w-full overflow-hidden">


        <div className="absolute inset-0 z-0">
          <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
            <color attach="background" args={['#050505']} />
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <BrainHero color={currentEmotion.color} />
            <BackgroundParticles color={currentEmotion.color} />
            <OrbitControls enableZoom={false} enablePan={false} />
            <Environment preset="city" />
          </Canvas>
        </div>

        {/* TYTUŁ EMOCJI */}
        <div className="relative z-10 h-full flex flex-col items-center justify-start pt-12 pointer-events-none">
          <AnimatePresence mode='wait'>
            <motion.div
              key={currentEmotion.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center"
            >
              <h1
                className="text-7xl md:text-9xl font-bold tracking-tighter text-transparent bg-clip-text pb-4"
                style={{
                  backgroundImage: `linear-gradient(to bottom, #ffffff 30%, ${currentEmotion.color})`,
                  textShadow: `0 0 25px ${currentEmotion.color}`
                }}
              >
                {currentEmotion.name}
              </h1>
              <p className="mt-2 text-xl text-gray-400 tracking-widest uppercase">
                Energy: <span className="text-white">{currentEmotion.energy}</span> | Mood: <span className="text-white">{currentEmotion.mood}</span>
              </p>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* PRZYCISK LOGOWANIA (JEŚLI NIEZALOGOWANY) */}
        {!token && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
            <button onClick={handleLogin}
              className="px-8 py-4 bg-green-500 text-black font-bold rounded-full text-xl hover:scale-105 transition shadow-xl pointer-events-auto">
              CONNECT SPOTIFY
            </button>
          </div>
        )}

        {/* LOGOUT BUTTON (TOP RIGHT) */}
        {token && (
          <div className="absolute top-4 right-4 z-50">
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-500/20 hover:bg-red-500/40 text-red-200 text-sm font-medium rounded-full backdrop-blur-md border border-red-500/30 transition-all"
            >
              DISCONNECT
            </button>
          </div>
        )}
      </div>

      {/* 2. SEKCJA WYBORU PLAYLISTY (DÓŁ) */}
      {token && (
        <div className="relative z-10 w-full min-h-[60vh] bg-black flex flex-col items-center pt-8 pb-32">

          {/* TŁO Z GWIAZDAMI (DÓŁ) */}
          <div className="absolute inset-0 z-0">
            <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
              <color attach="background" args={['#000000']} />
              <BackgroundParticles color={currentEmotion.color} />
            </Canvas>
          </div>

          <div className="relative z-10 w-full flex flex-col items-center">
            <h2 className="text-gray-400 text-sm uppercase tracking-widest mb-6">
              Recommended Soundscapes for {currentEmotion.name}
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 px-4 max-w-5xl w-full">
              {currentEmotion.playlists.map((playlist) => (
                <button
                  key={playlist.uri}
                  onClick={() => {
                    setActiveUri(playlist.uri);
                    setPlay(true);
                  }}
                  className={`
                    relative overflow-hidden group p-6 rounded-2xl border transition-all duration-300 text-left h-full
                    ${activeUri === playlist.uri
                      ? `border-white bg-white/10 scale-105 shadow-[0_0_30px_rgba(255,255,255,0.1)]`
                      : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/30'}
                  `}
                  style={{ borderColor: activeUri === playlist.uri ? currentEmotion.color : '' }}
                >
                  <div className="flex flex-col h-full justify-between relative z-10">
                    <div className="mb-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="text-xl font-bold text-white mb-1">{playlist.title}</h3>
                          <p className="text-xs text-gray-400">Curated for {currentEmotion.energy} energy</p>
                        </div>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${activeUri === playlist.uri ? 'bg-green-500 text-black' : 'bg-white/10 text-white'}`}>
                          {activeUri === playlist.uri ? "▶" : "•"}
                        </div>
                      </div>
                    </div>

                    {/* Song List Preview */}
                    <div className="space-y-2 mt-2">
                      {(playlistTracks[playlist.uri] || []).slice(0, 3).map((song, idx) => (
                        <div key={idx} className="flex items-center justify-between text-xs group/song">
                          <span className="text-gray-300 font-medium group-hover/song:text-white transition-colors truncate max-w-[70%]">
                            {song.title}
                          </span>
                          <span className="text-gray-500 group-hover/song:text-gray-400 transition-colors truncate max-w-[25%]">
                            {song.artist}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Glow effect based on emotion color */}
                  <div
                    className="absolute -right-4 -bottom-4 w-24 h-24 blur-2xl opacity-20 rounded-full transition-colors duration-500 pointer-events-none"
                    style={{ backgroundColor: currentEmotion.color }}
                  />
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 3. ODTWARZACZ (FIXED) */}
      {token && (
        <div className="fixed bottom-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-xl border-t border-white/10 p-2">
          {playerError && (
            <div className="bg-red-500/20 text-red-200 px-4 py-2 text-xs text-center mb-2 rounded border border-red-500/30">
              Spotify Error: {playerError}
            </div>
          )}
          <SpotifyPlayer
            token={token}
            name="Brain Tunes"
            uris={activeUri ? [activeUri] : []}
            play={play}
            initialVolume={0.5}
            persistDeviceSelection
            autoPlay={true}
            magnifySliderOnHover={true}
            callback={state => {
              if (state.isPlaying !== play) {
                setPlay(state.isPlaying);
              }

              if (state.error) {
                console.error("Spotify Player Error:", state.error);
                setPlayerError(state.error);
                if (state.errorType === 'authentication_error') {
                  handleLogout();
                }
              }
            }}
            styles={{
              activeColor: '#1db954',
              bgColor: 'transparent',
              color: '#fff',
              loaderColor: '#fff',
              sliderColor: '#1db954',
              sliderHandleColor: '#fff',
              trackArtistColor: '#ccc',
              trackNameColor: '#fff',
              height: '60px',
            }}
          />
        </div>
      )}
    </div>
  );
}

export default App;
