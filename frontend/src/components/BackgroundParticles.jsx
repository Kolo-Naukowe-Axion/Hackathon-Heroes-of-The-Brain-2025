import React from 'react';
import { Sparkles } from '@react-three/drei';

export function BackgroundParticles({ color }) {
    return (
        <Sparkles
            count={200}
            scale={12}
            size={4}
            speed={0.4}
            opacity={0.7}
            color={color}
        />
    );
}
