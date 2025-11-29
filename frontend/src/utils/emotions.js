export const emotions = [
    {
        name: 'neutral',
        color: '#9CA3AF', // Gray
        mood: 'neutral',
        energy: 'balanced',
        playlists: [
            {
                title: 'Balanced Flow',
                uri: 'spotify:playlist:0YcFmi7AedUHNL7SPOYCuP',
                songs: []
            }
        ]
    },
    {
        name: 'calm',
        color: '#34D399', // Green/Teal
        mood: 'pleasant',
        energy: 'low',
        playlists: [
            {
                title: 'Peaceful Piano',
                uri: 'spotify:playlist:37i9dQZF1DX4sWSpwq3LiO',
                songs: []
            },
            {
                title: 'Calm Vibes',
                uri: 'spotify:playlist:37i9dQZF1DX5bjCEbRU4SJ',
                songs: []
            },
            {
                title: 'Serenity Now',
                uri: 'spotify:playlist:0K7BHHVH6zRLuwZ9jOEncB',
                songs: []
            }
        ]
    },
    {
        name: 'happy',
        color: '#FDE047', // Yellow
        mood: 'pleasant',
        energy: 'high',
        playlists: [
            {
                title: 'Joyful Mix',
                uri: 'spotify:playlist:2mu4kG7W1LVjDh8SsxZBLF',
                songs: []
            },
            {
                title: 'Upbeat Energy',
                uri: 'spotify:playlist:5nRE43di91Z1APlH0z5rJl',
                songs: []
            },
            {
                title: 'Good Times',
                uri: 'spotify:playlist:1ynJ63k9UYnKQWH2Q0KMml',
                songs: []
            }
        ]
    },
    {
        name: 'sad',
        color: '#60A5FA', // Blue
        mood: 'unpleasant',
        energy: 'low',
        playlists: [
            {
                title: 'Comforting Sounds',
                uri: 'spotify:playlist:37i9dQZF1DXaImRpG7HXqp',
                songs: []
            },
            {
                title: 'Gentle Embrace',
                uri: 'spotify:playlist:5SQRulT1igDqNMuyRahwJQ',
                songs: []
            },
            {
                title: 'Healing Rhythms',
                uri: 'spotify:playlist:7pfUDjBvoR5IhurQ42CCgy',
                songs: []
            }
        ]
    },
    {
        name: 'angry',
        color: '#F87171', // Red
        mood: 'unpleasant',
        energy: 'high',
        playlists: [
            {
                title: 'Cool Down',
                uri: 'spotify:playlist:4FMOBw7eopNczgfzspCvIP',
                songs: []
            },
            {
                title: 'Release Tension',
                uri: 'spotify:playlist:2aM2P6Yso6qdXSFC3hFumC',
                songs: []
            },
            {
                title: 'Find Balance',
                uri: 'spotify:playlist:37i9dQZF1DXcr2UzLGERUU',
                songs: []
            }
        ]
    },
];

export const getEmotionColor = (emotionName) => {
    const emotion = emotions.find(e => e.name.toLowerCase() === emotionName.toLowerCase());
    return emotion ? emotion.color : '#E5E7EB';
};
