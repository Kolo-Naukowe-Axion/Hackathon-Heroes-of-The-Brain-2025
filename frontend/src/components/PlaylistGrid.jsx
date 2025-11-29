import React from 'react';

const playlists = [
    {
        id: 1,
        title: 'Chill Vibes',
        description: 'Relax and unwind.',
        songs: [
            { title: 'Weightless', artist: 'Marconi Union' },
            { title: 'Clair de Lune', artist: 'Claude Debussy' },
            { title: 'Gymnopédie No.1', artist: 'Erik Satie' },
        ]
    },
    {
        id: 2,
        title: 'Energy Boost',
        description: 'Get pumped up!',
        songs: [
            { title: 'Eye of the Tiger', artist: 'Survivor' },
            { title: 'Stronger', artist: 'Kanye West' },
            { title: 'Can\'t Hold Us', artist: 'Macklemore' },
        ]
    },
    {
        id: 3,
        title: 'Deep Focus',
        description: 'Concentrate on your work.',
        songs: [
            { title: 'Time', artist: 'Hans Zimmer' },
            { title: 'Cornfield Chase', artist: 'Hans Zimmer' },
            { title: 'Experience', artist: 'Ludovico Einaudi' },
        ]
    },
];

export function PlaylistGrid({ color }) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl px-4 z-10 relative mt-10">
            {playlists.map((playlist) => (
                <div
                    key={playlist.id}
                    className="group relative overflow-hidden rounded-xl border transition-all duration-500 hover:-translate-y-1"
                    style={{
                        borderColor: color,
                        backgroundColor: 'rgba(0, 0, 0, 0.4)',
                        boxShadow: `0 0 10px ${color}20`, // Subtle glow initially
                    }}
                >
                    {/* Hover Glow Effect */}
                    <div
                        className="absolute inset-0 opacity-0 group-hover:opacity-20 transition-opacity duration-500 pointer-events-none"
                        style={{ backgroundColor: color }}
                    />

                    <div className="p-6 relative z-10 flex flex-col h-full">
                        {/* Header with "Circuit" decoration */}
                        <div className="flex justify-between items-start mb-6">
                            <div className="flex space-x-1">
                                <div className="w-1 h-1 rounded-full bg-white/50" />
                                <div className="w-1 h-1 rounded-full bg-white/50" />
                                <div className="w-1 h-1 rounded-full bg-white/50" />
                            </div>
                            <div
                                className="text-xs font-mono tracking-widest uppercase opacity-60"
                                style={{ color: color }}
                            >
                                NO.{playlist.id.toString().padStart(2, '0')}
                            </div>
                        </div>

                        {/* Visualizer Placeholder */}
                        <div className="flex-grow flex items-center justify-center mb-6 h-32 w-full">
                            <div className="flex items-end space-x-1 h-16">
                                {[...Array(8)].map((_, i) => (
                                    <div
                                        key={i}
                                        className="w-2 bg-white/80 rounded-t-sm animate-pulse"
                                        style={{
                                            height: `${Math.random() * 100}%`,
                                            animationDelay: `${i * 0.1}s`,
                                            backgroundColor: color
                                        }}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* Content */}
                        <div className="flex-grow">
                            <h3 className="text-2xl font-light mb-2 text-white tracking-wide">
                                {playlist.title}
                            </h3>
                            <p className="text-sm text-gray-400 font-mono leading-relaxed mb-4">
                                {playlist.description}
                            </p>

                            {/* Song List */}
                            <div className="space-y-2">
                                {playlist.songs.map((song, index) => (
                                    <div key={index} className="flex items-center justify-between text-xs group/song">
                                        <span className="text-gray-300 font-medium group-hover/song:text-white transition-colors">
                                            {song.title}
                                        </span>
                                        <span className="text-gray-500 group-hover/song:text-gray-400 transition-colors">
                                            {song.artist}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Bottom "Wire" decoration */}
                        <div className="mt-6 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                    </div>
                </div>
            ))}
        </div>
    );
}
