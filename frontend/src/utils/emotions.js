export const emotions = [
    {
        name: 'neutral',
        color: '#9CA3AF', // Gray
        mood: 'neutral',
        energy: 'balanced',
        playlists: [
            {
                title: 'Instrumental',
                uri: 'spotify:playlist:0YcFmi7AedUHNL7SPOYCuP',
                songs: []
            },
            {
                title: 'Standard',
                uri: 'spotify:playlist:34BelWDaOFBBqF0tWoDeuD',
                songs: []
            },
            {
                title: 'For Kids',
                uri: 'spotify:playlist:1ynJ63k9UYnKQWH2Q0KMml',
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
                title: 'Instrumental',
                uri: 'spotify:playlist:37i9dQZF1DX4sWSpwq3LiO', // NOTE: This playlist returns 404 - needs to be replaced with a valid playlist URI
                songs: []
            },
            {
                title: 'Standard',
                uri: 'spotify:playlist:5PoCStl1p2KypDNfHjpM9j', // Calm Hits (User Curated)
                songs: []
            },
            {
                title: 'For Kids',
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
                title: 'Instrumental',
                uri: 'spotify:playlist:2mu4kG7W1LVjDh8SsxZBLF',
                songs: []
            },
            {
                title: 'Standard',
                uri: 'spotify:playlist:5nRE43di91Z1APlH0z5rJl',
                songs: []
            },
            {
                title: 'For Kids',
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
                title: 'Instrumental',
                uri: 'spotify:playlist:2SYQZPYbWDrKvDos8bpJHz', // Sad Instrumental Jazz
                songs: []
            },
            {
                title: 'Standard',
                uri: 'spotify:playlist:5SQRulT1igDqNMuyRahwJQ',
                songs: []
            },
            {
                title: 'For Kids',
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
                title: 'Instrumental',
                uri: 'spotify:playlist:4FMOBw7eopNczgfzspCvIP',
                songs: []
            },
            {
                title: 'Standard',
                uri: 'spotify:playlist:2aM2P6Yso6qdXSFC3hFumC',
                songs: []
            },
            {
                title: 'For Kids',
                uri: 'spotify:playlist:37i9dQZF1DXaImRpG7HXqp', // NOTE: This playlist returns 404 - needs to be replaced with a valid playlist URI
                songs: []
            }
        ]
    },
];

export const getEmotionColor = (emotionName) => {
    const emotion = emotions.find(e => e.name.toLowerCase() === emotionName.toLowerCase());
    return emotion ? emotion.color : '#E5E7EB';
};
