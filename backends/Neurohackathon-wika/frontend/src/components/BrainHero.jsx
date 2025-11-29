import React, { useRef, useMemo } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';
import { BrainMaterial } from './BrainMaterial';

export function BrainHero({ color }) {
    const meshRef = useRef();
    const materialRef = useRef();
    const obj = useLoader(OBJLoader, '/models/ultra_simplify_brain.obj');

    const targetColor = useMemo(() => new THREE.Color(color), [color]);

    const scene = useMemo(() => {
        const clone = obj.clone();
        // We will apply the material in the render loop or via the primitive, 
        // but for OBJLoader we often need to traverse and set material.
        // However, since we want to use the custom material component, 
        // we might need to construct the mesh manually or just apply the material to the children.

        // Actually, let's just use the geometry from the OBJ and render it inside a <mesh>
        // But OBJ might have multiple children. Let's traverse.
        let geometry = null;
        clone.traverse((child) => {
            if (child.isMesh) {
                geometry = child.geometry;
            }
        });
        return geometry;
    }, [obj]);

    useFrame((state, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * 0.05;
        }
        if (materialRef.current) {
            materialRef.current.uTime += delta;
            // Smoothly interpolate color
            materialRef.current.uColor.lerp(targetColor, delta * 2);
        }
    });

    if (!scene) return null;

    return (
        <mesh
            ref={meshRef}
            geometry={scene}
            position={[0, -1, 0]}
            scale={[0.75, 0.75, 0.75]}
            rotation={[0, 0, 0]}
        >
            {/* @ts-ignore */}
            <brainMaterial
                ref={materialRef}
                wireframe={true}
                transparent={true}
                side={THREE.DoubleSide}
            />
        </mesh>
    );
}
