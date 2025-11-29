import React from 'react';

const playlists = [
    { id: 1, title: 'Chill Vibes', description: 'Relax and unwind.' },
    { id: 2, title: 'Energy Boost', description: 'Get pumped up!' },
    { id: 3, title: 'Deep Focus', description: 'Concentrate on your work.' },
];

export function PlaylistGrid({ color }) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl px-4 z-10 relative mt-10">
            {playlists.map((playlist) => (
                <div
                    key={playlist.id}
                    className="backdrop-blur-md bg-white/10 rounded-xl p-6 border-2 transition-colors duration-1000 flex flex-col items-center text-center shadow-lg hover:bg-white/20 cursor-pointer"
                    style={{ borderColor: color }}
                >
                    <div className="w-full h-40 bg-black/30 rounded-lg mb-4 flex items-center justify-center">
                        <span className="text-4xl">🎵</span>
                    </div>
                    <h3 className="text-xl font-bold mb-2 text-white">{playlist.title}</h3>
                    <p className="text-gray-300">{playlist.description}</p>
                </div>
            ))}
        </div>
    );
}
