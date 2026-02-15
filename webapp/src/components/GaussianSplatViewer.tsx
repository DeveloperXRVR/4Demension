"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Loader2,
  Maximize2,
  Minimize2,
  RotateCcw,
  Download,
  Info,
  Grid3x3,
  Sun,
  Eye,
} from "lucide-react";

interface GaussianSplatViewerProps {
  modelUrl: string;
  jobId: string;
}

export default function GaussianSplatViewer({
  modelUrl,
  jobId,
}: GaussianSplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointCloudRef = useRef<THREE.Points | null>(null);
  const animFrameRef = useRef<number>(0);

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [pointCount, setPointCount] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.01,
      1000
    );
    camera.position.set(3, 2, 3);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.5;
    controls.maxDistance = 50;
    controls.maxPolarAngle = Math.PI;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);
    const hemisphereLight = new THREE.HemisphereLight(0x8888ff, 0x444422, 0.4);
    scene.add(hemisphereLight);

    // Grid
    const gridHelper = new THREE.GridHelper(10, 20, 0x333333, 0x222222);
    gridHelper.name = "grid";
    scene.add(gridHelper);

    // Axes
    const axesHelper = new THREE.AxesHelper(2);
    axesHelper.name = "axes";
    scene.add(axesHelper);

    // Load model
    loadSplatModel(scene, modelUrl);

    // Animation loop
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("resize", handleResize);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelUrl]);

  async function loadSplatModel(scene: THREE.Scene, url: string) {
    setLoading(true);
    setLoadError(null);

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to load model: ${response.status}`);

      const arrayBuffer = await response.arrayBuffer();
      const dataView = new DataView(arrayBuffer);

      // Parse the splat data
      // Each point: pos(3f=12) + scale(3f=12) + color(4B=4) + rot(1f=4) = 32 bytes
      const bytesPerPoint = 32;
      const numPoints = Math.floor(arrayBuffer.byteLength / bytesPerPoint);

      if (numPoints === 0) {
        throw new Error("No points found in model file");
      }

      setPointCount(numPoints);

      const positions = new Float32Array(numPoints * 3);
      const colors = new Float32Array(numPoints * 3);
      const sizes = new Float32Array(numPoints);

      for (let i = 0; i < numPoints; i++) {
        const offset = i * bytesPerPoint;

        // Position
        positions[i * 3] = dataView.getFloat32(offset, true);
        positions[i * 3 + 1] = dataView.getFloat32(offset + 4, true);
        positions[i * 3 + 2] = dataView.getFloat32(offset + 8, true);

        // Scale (average for point size)
        const sx = dataView.getFloat32(offset + 12, true);
        const sy = dataView.getFloat32(offset + 16, true);
        const sz = dataView.getFloat32(offset + 20, true);
        sizes[i] = (Math.abs(sx) + Math.abs(sy) + Math.abs(sz)) / 3 * 100;

        // Color
        colors[i * 3] = dataView.getUint8(offset + 24) / 255;
        colors[i * 3 + 1] = dataView.getUint8(offset + 25) / 255;
        colors[i * 3 + 2] = dataView.getUint8(offset + 26) / 255;
      }

      // Create point cloud geometry
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

      // Custom shader material for splat-like rendering
      const material = new THREE.ShaderMaterial({
        uniforms: {
          pointMultiplier: {
            value: window.innerHeight / (2.0 * Math.tan((60 * Math.PI) / 360)),
          },
        },
        vertexShader: `
          attribute float size;
          varying vec3 vColor;
          uniform float pointMultiplier;
          void main() {
            vColor = color;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mvPosition;
            gl_PointSize = size * pointMultiplier / (-mvPosition.z);
            gl_PointSize = clamp(gl_PointSize, 1.0, 64.0);
          }
        `,
        fragmentShader: `
          varying vec3 vColor;
          void main() {
            // Gaussian splat-like soft circle
            vec2 center = gl_PointCoord - vec2(0.5);
            float dist = length(center);
            float alpha = exp(-8.0 * dist * dist);
            if (alpha < 0.01) discard;
            gl_FragColor = vec4(vColor, alpha);
          }
        `,
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });

      const points = new THREE.Points(geometry, material);
      points.name = "splatCloud";
      scene.add(points);
      pointCloudRef.current = points;

      // Center camera on the point cloud
      geometry.computeBoundingSphere();
      if (geometry.boundingSphere) {
        const center = geometry.boundingSphere.center;
        const radius = geometry.boundingSphere.radius;
        if (controlsRef.current && cameraRef.current) {
          controlsRef.current.target.copy(center);
          cameraRef.current.position.set(
            center.x + radius * 1.5,
            center.y + radius,
            center.z + radius * 1.5
          );
          controlsRef.current.update();
        }
      }

      setLoading(false);
    } catch (err) {
      console.error("Model load error:", err);
      setLoadError(err instanceof Error ? err.message : "Failed to load model");
      setLoading(false);
    }
  }

  const resetCamera = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(3, 2, 3);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const toggleGrid = () => {
    if (!sceneRef.current) return;
    const grid = sceneRef.current.getObjectByName("grid");
    if (grid) grid.visible = !grid.visible;
    setShowGrid(!showGrid);
  };

  const toggleAxes = () => {
    if (!sceneRef.current) return;
    const axes = sceneRef.current.getObjectByName("axes");
    if (axes) axes.visible = !axes.visible;
    setShowAxes(!showAxes);
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = modelUrl;
    a.download = `4demension-${jobId.slice(0, 8)}.splat`;
    a.click();
  };

  return (
    <div className="w-full h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/90 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-violet-400" />
          <span className="text-sm font-medium">3D Gaussian Splat Viewer</span>
          {pointCount > 0 && (
            <span className="text-xs text-zinc-500 ml-2">
              {pointCount.toLocaleString()} points
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={toggleGrid}
            className={`p-2 rounded-lg transition-colors ${
              showGrid
                ? "bg-violet-600/20 text-violet-400"
                : "hover:bg-zinc-800 text-zinc-500"
            }`}
            title="Toggle Grid"
          >
            <Grid3x3 className="w-4 h-4" />
          </button>
          <button
            onClick={toggleAxes}
            className={`p-2 rounded-lg transition-colors ${
              showAxes
                ? "bg-violet-600/20 text-violet-400"
                : "hover:bg-zinc-800 text-zinc-500"
            }`}
            title="Toggle Axes"
          >
            <Sun className="w-4 h-4" />
          </button>
          <button
            onClick={resetCamera}
            className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 transition-colors"
            title="Reset Camera"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={handleDownload}
            className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 transition-colors"
            title="Download Model"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 transition-colors"
            title="Fullscreen"
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Viewer */}
      <div ref={containerRef} className="flex-1 relative min-h-0 splat-viewer">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/80 z-10">
            <Loader2 className="w-10 h-10 animate-spin text-violet-400 mb-4" />
            <p className="text-sm text-zinc-400">Loading 3D model...</p>
          </div>
        )}
        {loadError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/80 z-10">
            <Info className="w-10 h-10 text-red-400 mb-4" />
            <p className="text-sm text-red-400 mb-2">Failed to load model</p>
            <p className="text-xs text-zinc-500">{loadError}</p>
          </div>
        )}
      </div>

      {/* Controls hint */}
      <div className="px-4 py-2 bg-zinc-900/90 border-t border-zinc-800 flex items-center justify-center gap-6 text-xs text-zinc-600">
        <span>🖱️ Left: Rotate</span>
        <span>🖱️ Right: Pan</span>
        <span>🖱️ Scroll: Zoom</span>
      </div>
    </div>
  );
}
