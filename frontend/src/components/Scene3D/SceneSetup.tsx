import React, { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { CoordinateSystem } from './CoordinateSystem';

export const SceneSetup: React.FC = () => {
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(16, 14, 16);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return (
    <>
      {/* Balanced ambient & directional illumination */}
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />

      {/* OrbitControls with smooth damping */}
      <OrbitControls
        enableDamping={true}
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={60}
        makeDefault
      />

      {/* Minimalist architectural grid floor */}
      <Grid
        position={[0, -4, 0]}
        cellSize={1}
        cellColor="#18181b"
        sectionSize={5}
        sectionColor="#27272a"
        fadeDistance={50}
        infiniteGrid={true}
      />

      {/* 3D Coordinate System with labeled axes, origin marker, and orientation gizmo */}
      <CoordinateSystem />
    </>
  );
};
