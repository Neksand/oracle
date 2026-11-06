import React, { useMemo } from 'react';
import { Text, Line, GizmoHelper, GizmoViewport } from '@react-three/drei';

interface DistanceRingProps {
  radius: number;
  label: string;
}

const DistanceRing: React.FC<DistanceRingProps> = ({ radius, label }) => {
  const points = useMemo(() => {
    const pts: [number, number, number][] = [];
    const segments = 64;
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      pts.push([Math.cos(theta) * radius, 0, Math.sin(theta) * radius]);
    }
    return pts;
  }, [radius]);

  return (
    <group>
      <Line
        points={points}
        color="#27272a"
        lineWidth={1}
        dashed={true}
        dashScale={2}
        dashSize={0.5}
        gapSize={0.3}
      />
      <Text
        position={[radius + 0.3, 0, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.35}
        color="#71717a"
        anchorX="left"
        anchorY="middle"
      >
        {label}
      </Text>
    </group>
  );
};

export const CoordinateSystem: React.FC = () => {
  return (
    <group>
      {/* 1. Origin Marker (0,0,0) */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      <Text
        position={[0.45, 0.55, 0]}
        fontSize={0.38}
        color="#ffffff"
        anchorX="left"
        anchorY="bottom"
      >
        S₀ · current level
      </Text>

      {/* 2. Y-Axis (Trend Direction / Drift) */}
      <Line
        points={[[0, -12, 0], [0, 14, 0]]}
        color="#38bdf8"
        lineWidth={1.5}
      />
      <mesh position={[0, 14, 0]}>
        <coneGeometry args={[0.25, 0.6, 16]} />
        <meshBasicMaterial color="#38bdf8" />
      </mesh>
      <Text
        position={[0.5, 14.8, 0]}
        fontSize={0.48}
        color="#38bdf8"
        anchorX="left"
        anchorY="bottom"
      >
        +Y · Capital expansion (μ{'>'} 0)
      </Text>
      <Text
        position={[0.5, -12.8, 0]}
        fontSize={0.42}
        color="#f87171"
        anchorX="left"
        anchorY="top"
      >
        −Y · Capital contraction (μ{'<'} 0)
      </Text>

      {/* 3. X-Axis (Lateral Uncertainty) */}
      <Line
        points={[[-14, 0, 0], [14, 0, 0]]}
        color="#71717a"
        lineWidth={1}
      />
      <mesh position={[14, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.2, 0.5, 16]} />
        <meshBasicMaterial color="#a1a1aa" />
      </mesh>
      <Text
        position={[14.6, 0.4, 0]}
        fontSize={0.44}
        color="#a1a1aa"
        anchorX="left"
        anchorY="middle"
      >
        +X · Idiosyncratic shock (σ)
      </Text>

      {/* 4. Z-Axis (Depth Uncertainty) */}
      <Line
        points={[[0, 0, -14], [0, 0, 14]]}
        color="#71717a"
        lineWidth={1}
      />
      <mesh position={[0, 0, 14]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.2, 0.5, 16]} />
        <meshBasicMaterial color="#a1a1aa" />
      </mesh>
      <Text
        position={[0.5, 0.4, 14.6]}
        fontSize={0.44}
        color="#a1a1aa"
        anchorX="left"
        anchorY="middle"
      >
        +Z · Residual noise (σ)
      </Text>

      {/* 5. Concentric Radial Distance Markers on ground plane */}
      <DistanceRing radius={4} label="r = 4" />
      <DistanceRing radius={8} label="r = 8" />
      <DistanceRing radius={12} label="r = 12" />

      {/* 6. Interactive 3D Orientation Gizmo in bottom-right corner */}
      <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
        <GizmoViewport
          axisColors={['#f43f5e', '#38bdf8', '#a855f7']}
          labelColor="#ffffff"
          hideNegativeAxes={false}
        />
      </GizmoHelper>
    </group>
  );
};
