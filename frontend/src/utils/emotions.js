export const emotions = [
    {
        name: 'neutral',
        color: '#9CA3AF', // Gray
        mood: 'neutral',
        energy: 'balanced',
        playlists: [
            {
                title: 'Deep Focus',
                uri: 'spotify:playlist:37i9dQZF1DWZeKCadgRdKQ',
                songs: []
            },
            {
                title: 'Lo-Fi Beats',
                uri: 'spotify:playlist:37i9dQZF1DWWQRwui0ExPn',
                songs: []
            },
            {
                title: 'Brain Food',
                uri: 'spotify:playlist:37i9dQZF1DWXLeA8Omikj7',
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
                title: 'Sleep',
                uri: 'spotify:playlist:37i9dQZF1DWZd79rJ6a7lp',
                songs: []
            },
            {
                title: 'Nature Sounds',
                uri: 'spotify:playlist:37i9dQZF1DX4wta20PHgwo',
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
                title: 'Mood Booster',
                uri: 'spotify:playlist:37i9dQZF1DX3rxVfibe1L0',
                songs: []
            },
            {
                title: 'Happy Hits',
                uri: 'spotify:playlist:37i9dQZF1DXdPec7aLTmlC',
                songs: []
            },
            {
                title: 'Good Vibes',
                uri: 'spotify:playlist:37i9dQZF1DWYBO1MoTDhZI',
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
                title: 'Life Sucks',
                uri: 'spotify:playlist:37i9dQZF1DX3YSRoSdA634',
                songs: []
            },
            {
                title: 'Sad Songs',
                uri: 'spotify:playlist:37i9dQZF1DX7qK8ma5wgG1',
                songs: []
            },
            {
                title: 'Broken Heart',
                uri: 'spotify:playlist:37i9dQZF1DX889U0CL85jj',
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
                title: 'Rage Beats',
                uri: 'spotify:playlist:37i9dQZF1DX3oM43CtKnRV',
                songs: []
            },
            {
                title: 'Rock Hard',
                uri: 'spotify:playlist:37i9dQZF1DWXRqgorJj26U',
                songs: []
            },
            {
                title: 'Metal Essentials',
                uri: 'spotify:playlist:37i9dQZF1DWWOaP4H0w5b0',
                songs: []
            }
        ]
    },
];

export const getEmotionColor = (emotionName) => {
    const emotion = emotions.find(e => e.name.toLowerCase() === emotionName.toLowerCase());
    return emotion ? emotion.color : '#E5E7EB';
};
