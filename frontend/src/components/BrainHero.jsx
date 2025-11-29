import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { TorusKnot } from '@react-three/drei';
import * as THREE from 'three';

export function BrainHero({ color }) {
    const meshRef = useRef();
    const materialRef = useRef();
    const targetColor = new THREE.Color(color);

    useFrame((state, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.x += delta * 0.2;
            meshRef.current.rotation.y += delta * 0.3;
        }
        if (materialRef.current) {
            materialRef.current.color.lerp(targetColor, delta * 2);
            materialRef.current.emissive.lerp(targetColor, delta * 2);
        }
    });

    return (
        <TorusKnot args={[1, 0.3, 128, 32]} ref={meshRef} position={[0, 0, 0]}>
            <meshStandardMaterial
                ref={materialRef}
                color={color} // Initial color
                emissive={color}
                emissiveIntensity={0.5}
                roughness={0.2}
                metalness={0.8}
                wireframe={true}
            />
        </TorusKnot>
    );
}
