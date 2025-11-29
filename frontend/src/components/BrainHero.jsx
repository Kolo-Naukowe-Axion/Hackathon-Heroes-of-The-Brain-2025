import React, { useRef } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';

export function BrainHero({ color }) {
    const meshRef = useRef();
    const obj = useLoader(OBJLoader, '/models/ultra_simplify_brain.obj');

    // Create a material that we will reuse
    const material = React.useMemo(() => {
        return new THREE.MeshStandardMaterial({
            color: new THREE.Color(color),
            emissive: new THREE.Color(color),
            emissiveIntensity: 0.5,
            roughness: 0.2,
            metalness: 0.8,
            wireframe: true
        });
    }, []); // Created once

    const targetColor = new THREE.Color(color);

    const scene = React.useMemo(() => {
        const clone = obj.clone();
        clone.traverse((child) => {
            if (child.isMesh) {
                child.material = material;
            }
        });
        return clone;
    }, [obj, material]);

    useFrame((state, delta) => {
        if (meshRef.current) {
            // Slow "idle" rotation
            meshRef.current.rotation.y += delta * 0.05;
        }
        // Animate material
        material.color.lerp(targetColor, delta * 2);
        material.emissive.lerp(targetColor, delta * 2);
    });

    return (
        <primitive
            object={scene}
            ref={meshRef}
            position={[0, -1, 0]}
            scale={[0.75, 0.75, 0.75]}
            rotation={[0, 0, 0]}
        />
    );
}
