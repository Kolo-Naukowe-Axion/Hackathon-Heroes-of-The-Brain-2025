export const emotions = [
    { name: 'Neutral', color: '#F3F4F6', mood: 'neutral', energy: 'balanced' }, // gray-100
    { name: 'Happy', color: '#FDE047', mood: 'good', energy: 'high' },   // yellow-300
    { name: 'Sad', color: '#60A5FA', mood: 'bad', energy: 'low' },     // blue-400
    { name: 'Angry', color: '#F87171', mood: 'bad', energy: 'high' },   // red-400
    { name: 'Calm', color: '#34D399', mood: 'good', energy: 'low' },    // emerald-400
];

export const getEmotionColor = (emotionName) => {
    const emotion = emotions.find(e => e.name === emotionName);
    return emotion ? emotion.color : '#E5E7EB';
};
