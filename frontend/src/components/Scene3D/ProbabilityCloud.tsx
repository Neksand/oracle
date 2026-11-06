import React, { useMemo, useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

export interface ProbabilityCloudProps {
  positions: number[];
  colors: number[];
  /** Version key: new run -> brand-new GPU buffers. Fixes stale-cloud bug. */
  runKey: string | number;
  pointSize?: number;
}

export const ProbabilityCloud: React.FC<ProbabilityCloudProps> = ({
  positions,
  colors,
  runKey,
  pointSize = 0.14,
}) => {
  // Fresh BufferGeometry per run: guarantees the GPU uploads new data even
  // when particle counts are identical (old code keyed only on array length,
  // so the cloud looked frozen when mu/sigma/S0 changed).
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    g.computeBoundingSphere();
    // Depend on runKey so an identical-length rerun still rebuilds buffers.
    void runKey;
    return g;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [positions, colors, runKey]);

  // Dispose the previous run's GPU buffers to avoid leaks.
  useEffect(() => () => geometry.dispose(), [geometry]);

  const ref = useRef<THREE.Points>(null);
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.035;
  });

  if (positions.length === 0 || colors.length === 0) {
    return null;
  }

  return (
    <points key={`cloud-${String(runKey)}`} ref={ref} geometry={geometry}>
      <PointMaterial
        size={pointSize}
        vertexColors={true}
        transparent={false}
        opacity={1.0}
        blending={THREE.NormalBlending}
        depthWrite={true}
        sizeAttenuation={true}
      />
    </points>
  );
};
