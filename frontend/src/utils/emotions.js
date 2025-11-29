export const emotions = [
    { name: 'Neutral', color: '#E5E7EB' }, // gray-200
    { name: 'Happy', color: '#FACC15' },   // yellow-400
    { name: 'Sad', color: '#1E3A8A' },     // blue-900
    { name: 'Angry', color: '#DC2626' },   // red-600
    { name: 'Calm', color: '#10B981' },    // emerald-500
];

export const getEmotionColor = (emotionName) => {
    const emotion = emotions.find(e => e.name === emotionName);
    return emotion ? emotion.color : '#E5E7EB';
};
