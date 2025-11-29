import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import SpotifyPlayer from 'react-spotify-web-playback';

import { emotions } from '../utils/emotions';
import { BrainHero } from '../components/BrainHero';
import { BackgroundParticles } from '../components/BackgroundParticles';
import { redirectToAuthCodeFlow, getAccessToken } from '../utils/auth';
import { fetchPlaylistTracks } from '../utils/spotify';

// --- KONFIGURACJA ---
const CLIENT_ID = "00ea48b1e2144865828b75c3d4746b7c"; // <--- PAMIĘTAJ O WPISANIU ID!
const REDIRECT_URI = "http://127.0.0.1:5173/";
const SCOPES = ["streaming", "user-read-email", "user-read-private", "user-read-playback-state", "user-modify-playback-state"];

function Home() {
  const [emotionIndex, setEmotionIndex] = useState(0); // 0 = Neutral
  const [token, setToken] = useState("");
  const [play, setPlay] = useState(false);
  const [activeUri, setActiveUri] = useState("");
  const [playlistTracks, setPlaylistTracks] = useState({});
  const [playerError, setPlayerError] = useState(null);
  const [isTokenValidated, setIsTokenValidated] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState(false); // false = Simulation, true = Connected

  // Logowanie PKCE
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");

    // Check for existing token
    const storedToken = window.localStorage.getItem("token");
    if (storedToken) {
      setToken(storedToken);
      // Validate token AND scopes by checking a playback endpoint
      fetch("https://api.spotify.com/v1/me/player/devices", {
        headers: { Authorization: `Bearer ${storedToken}` }
      }).then(res => {
        if (res.status === 401 || res.status === 403) {
          console.warn("Token expired or missing scopes, logging out...");
          handleLogout();
        } else {
          setIsTokenValidated(true);
        }
      }).catch(err => {
        console.error("Token validation failed:", err);
      });
    }

    if (code && !storedToken) {
      getAccessToken(CLIENT_ID, code, REDIRECT_URI).then((accessToken) => {
        if (accessToken) {
          window.localStorage.setItem("token", accessToken);
          setToken(accessToken);
          setIsTokenValidated(true); // Fresh token is valid
          window.history.replaceState({}, document.title, "/");
        }
      });
    }
  }, []);

  // WebSocket Connection for Emotion Updates
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
      console.log("Connected to Emotion WebSocket");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.emotion) {
          // Map backend emotion labels to frontend emotion indices
          // Backend labels: "NEGATIVE", "NEUTRAL", "POSITIVE"
          // Frontend indices: 0=Neutral, 1=Positive, 2=Negative
          let newIndex = 0;
          switch (data.emotion.toUpperCase()) {
            case 'NEUTRAL': newIndex = 0; break;
            case 'POSITIVE': newIndex = 1; break;
            case 'NEGATIVE': newIndex = 2; break;
            default: break; // Keep current if unknown
          }
          setEmotionIndex(newIndex);
        }
        if (data.is_connected !== undefined) {
          setConnectionStatus(data.is_connected);
        }
      } catch (e) {
        console.error("Error parsing WebSocket message:", e);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      ws.close();
    };
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
      // Global Spacebar Pause/Play
      if (event.code === 'Space') {
        event.preventDefault(); // Prevent scrolling
        setPlay(prev => !prev);
        return;
      }

      switch (event.key.toLowerCase()) {
        case 'n': setEmotionIndex(0); break; // Neutral
        case 'p': setEmotionIndex(1); break; // Positive
        case 'm': setEmotionIndex(2); break; // Negative (Minus)
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
  // Fetch tracks for ALL emotions' playlists when token is available
  // Fetch tracks for ALL emotions' playlists when token is available
  useEffect(() => {
    if (token) {
      emotions.forEach(emotion => {
        emotion.playlists.forEach(playlist => {
          // Only fetch if we haven't already
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
      });
    }
  }, [token]);

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-white/30">

      {/* 1. SEKCJA 3D (GÓRA) */}
      <div className="relative h-[70vh] w-full overflow-hidden">

        {/* HEADER (LOGO + DISCONNECT) */}
        <div className="absolute top-0 left-0 right-0 z-50 flex justify-between items-center p-6 pointer-events-none">
          {/* LOGO */}
          <div className="pointer-events-auto select-none flex flex-col">
            <img src="/logo_full.png" alt="Oscillate Logo" className="h-40 md:h-56 w-auto opacity-90" />
            <div className="ml-4 -mt-10">
              {connectionStatus ? (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/30 backdrop-blur-sm">
                  <span className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse"></span>
                  DEVICE CONNECTED
                </span>
              ) : (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-orange-500/20 text-orange-400 border border-orange-500/30 backdrop-blur-sm">
                  <span className="w-2 h-2 rounded-full bg-orange-500 mr-2"></span>
                  SIMULATION MODE
                </span>
              )}
            </div>
          </div>

          {/* DISCONNECT BUTTON */}
          <div className="pointer-events-auto">
            {token && (
              <button
                onClick={handleLogout}
                className="px-4 py-2 text-sm font-medium rounded-full backdrop-blur-md border transition-all"
                style={{
                  backgroundColor: `${currentEmotion.color}33`,
                  borderColor: `${currentEmotion.color}4d`,
                  color: currentEmotion.color
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = `${currentEmotion.color}66`}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = `${currentEmotion.color}33`}
              >
                DISCONNECT
              </button>
            )}
          </div>
        </div>


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
      {token && isTokenValidated && (
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
            magnifySliderOnHover={true}
            callback={state => {
              if (state.error) {
                console.error("Spotify Player Error:", state.error);
                setPlayerError(state.error);
                if (state.errorType === 'authentication_error') {
                  handleLogout();
                }
              }

              // Only sync state if the player is active and ready to avoid race conditions during loading
              if (state.isActive && state.status === 'READY') {
                if (state.isPlaying !== play) {
                  setPlay(state.isPlaying);
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

export default Home;
