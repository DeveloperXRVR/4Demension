"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Loader2,
  Maximize2,
  Minimize2,
  RotateCcw,
  Download,
  Info,
  Eye,
} from "lucide-react";

/* ── .splat binary parser ──────────────────────────────────────────────
   32 bytes per splat: pos(3×f32) + scale(3×f32) + rgba(4×u8) + rot(4×u8)
*/
function parseSplat(buffer: ArrayBuffer) {
  const bytesPerSplat = 32;
  const count = Math.floor(buffer.byteLength / bytesPerSplat);
  const view = new DataView(buffer);

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const alphas = new Float32Array(count);
  const sizes = new Float32Array(count);

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < count; i++) {
    const off = i * bytesPerSplat;
    const x = view.getFloat32(off, true);
    const y = view.getFloat32(off + 4, true);
    const z = view.getFloat32(off + 8, true);

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;

    const sx = view.getFloat32(off + 12, true);
    const sy = view.getFloat32(off + 16, true);
    const sz = view.getFloat32(off + 20, true);
    sizes[i] = (sx + sy + sz) / 3.0;

    colors[i * 3] = view.getUint8(off + 24) / 255;
    colors[i * 3 + 1] = view.getUint8(off + 25) / 255;
    colors[i * 3 + 2] = view.getUint8(off + 26) / 255;
    alphas[i] = view.getUint8(off + 27) / 255;
  }

  const center = new THREE.Vector3(
    (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2
  );
  const extent = new THREE.Vector3(
    maxX - minX, maxY - minY, maxZ - minZ
  );

  return { positions, colors, alphas, sizes, count, center, extent };
}

/* ── Custom shader for Gaussian-like splat points ──────────────────── */
const splatVertexShader = `
  attribute float size;
  attribute float alpha;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float uPixelRatio;
  uniform float uScaleMultiplier;

  void main() {
    vColor = color;
    vAlpha = alpha;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    // Size attenuation: scale splat size by distance
    float pointSize = size * uScaleMultiplier * uPixelRatio * (300.0 / -mvPosition.z);
    gl_PointSize = clamp(pointSize, 1.0, 64.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const splatFragmentShader = `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    // Gaussian falloff from center of point
    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    float r2 = dot(cxy, cxy);
    if (r2 > 1.0) discard;
    float gauss = exp(-4.0 * r2);
    gl_FragColor = vec4(vColor, vAlpha * gauss);
  }
`;

interface GaussianSplatViewerProps {
  modelUrl: string;
  jobId: string;
}

export default function GaussianSplatViewer({
  modelUrl,
  jobId,
}: GaussianSplatViewerProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    animId: number;
    defaultCamPos: THREE.Vector3;
    defaultTarget: THREE.Vector3;
  } | null>(null);
  const initedUrl = useRef<string | null>(null);

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [pointCount, setPointCount] = useState(0);

  useEffect(() => {
    const container = viewerContainerRef.current;
    if (!container) return;
    if (initedUrl.current === modelUrl) return;
    initedUrl.current = modelUrl;

    let disposed = false;

    const init = async () => {
      try {
        console.log("[SplatViewer] Fetching", modelUrl);
        const res = await fetch(modelUrl);
        if (!res.ok) throw new Error(`HTTP ${res.status} loading model`);
        const buffer = await res.arrayBuffer();
        console.log("[SplatViewer] Loaded", (buffer.byteLength / 1e6).toFixed(1), "MB");
        if (disposed) return;

        const data = parseSplat(buffer);
        console.log("[SplatViewer] Parsed", data.count, "splats, center:", data.center, "extent:", data.extent);

        // ── Three.js setup ──
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 600;

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x111111, 1);
        container.appendChild(renderer.domElement);

        const scene = new THREE.Scene();

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);
        // Position camera to see the whole scene
        const maxExtent = Math.max(data.extent.x, data.extent.y, data.extent.z, 1);
        const camDist = maxExtent * 1.5;
        const camPos = new THREE.Vector3(
          data.center.x + camDist * 0.5,
          data.center.y + camDist * 0.3,
          data.center.z + camDist * 0.8
        );
        camera.position.copy(camPos);
        camera.lookAt(data.center);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.copy(data.center);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.minDistance = 0.1;
        controls.maxDistance = maxExtent * 10;
        controls.update();

        // ── Build point cloud geometry ──
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(data.positions, 3));
        geometry.setAttribute("color", new THREE.Float32BufferAttribute(data.colors, 3));
        geometry.setAttribute("size", new THREE.Float32BufferAttribute(data.sizes, 1));
        geometry.setAttribute("alpha", new THREE.Float32BufferAttribute(data.alphas, 1));

        const material = new THREE.ShaderMaterial({
          vertexShader: splatVertexShader,
          fragmentShader: splatFragmentShader,
          uniforms: {
            uPixelRatio: { value: renderer.getPixelRatio() },
            uScaleMultiplier: { value: 1.0 },
          },
          vertexColors: true,
          transparent: true,
          depthWrite: false,
          blending: THREE.NormalBlending,
        });

        const points = new THREE.Points(geometry, material);
        scene.add(points);

        if (disposed) {
          renderer.dispose();
          geometry.dispose();
          material.dispose();
          return;
        }

        // ── Render loop ──
        let animId = 0;
        const animate = () => {
          animId = requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        };
        animate();

        // ── Resize handler ──
        const onResize = () => {
          const w = container.clientWidth;
          const h = container.clientHeight;
          if (w === 0 || h === 0) return;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        };
        const resizeObserver = new ResizeObserver(onResize);
        resizeObserver.observe(container);

        sceneRef.current = {
          renderer,
          scene,
          camera,
          controls,
          animId,
          defaultCamPos: camPos.clone(),
          defaultTarget: data.center.clone(),
        };

        setPointCount(data.count);
        setLoading(false);
        console.log("[SplatViewer] Rendering", data.count, "splats");
      } catch (err) {
        if (!disposed) {
          console.error("[SplatViewer] Error:", err);
          setLoadError(err instanceof Error ? err.message : "Failed to load 3D model");
          setLoading(false);
        }
      }
    };

    init();

    return () => {
      disposed = true;
      initedUrl.current = null;
      if (sceneRef.current) {
        cancelAnimationFrame(sceneRef.current.animId);
        sceneRef.current.controls.dispose();
        sceneRef.current.renderer.dispose();
        sceneRef.current.scene.traverse((obj) => {
          if (obj instanceof THREE.Points) {
            obj.geometry.dispose();
            if (obj.material instanceof THREE.Material) obj.material.dispose();
          }
        });
        sceneRef.current = null;
      }
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelUrl]);

  const resetCamera = useCallback(() => {
    const s = sceneRef.current;
    if (s) {
      s.camera.position.copy(s.defaultCamPos);
      s.controls.target.copy(s.defaultTarget);
      s.controls.update();
    }
  }, []);

  const toggleFullscreen = useCallback(() => {
    const el = wrapperRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const handleDownload = useCallback(() => {
    const a = document.createElement("a");
    a.href = modelUrl;
    a.download = `vlasovai-4dmap-${jobId.slice(0, 8)}.splat`;
    a.click();
  }, [modelUrl, jobId]);

  return (
    <div ref={wrapperRef} className="w-full h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/90 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-violet-400" />
          <span className="text-sm font-medium">3D просмотрщик</span>
          {pointCount > 0 && (
            <span className="text-xs text-zinc-500 ml-2">
              {pointCount.toLocaleString()} точек
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={resetCamera}
            className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 transition-colors"
            title="Сбросить камеру"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={handleDownload}
            className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 transition-colors"
            title="Скачать модель"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 transition-colors"
            title="Полный экран"
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Viewer area: wrapper is relative so overlay + viewer stack correctly */}
      <div className="flex-1 relative min-h-0" style={{ background: "#111" }}>
        {/* Imperative viewer container — React NEVER renders children here */}
        <div
          ref={viewerContainerRef}
          className="absolute inset-0"
          style={{ overflow: "hidden" }}
        />

        {/* React-managed overlays — completely separate from the viewer DOM */}
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/80 z-10 pointer-events-none">
            <Loader2 className="w-10 h-10 animate-spin text-violet-400 mb-4" />
            <p className="text-sm text-zinc-400">Загрузка 3D модели...</p>
          </div>
        )}
        {loadError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/80 z-10">
            <Info className="w-10 h-10 text-red-400 mb-4" />
            <p className="text-sm text-red-400 mb-2">Не удалось загрузить модель</p>
            <p className="text-xs text-zinc-500 max-w-md text-center">
              {loadError}
            </p>
          </div>
        )}
      </div>

      {/* Controls hint */}
      <div className="px-4 py-2 bg-zinc-900/90 border-t border-zinc-800 flex items-center justify-center gap-6 text-xs text-zinc-600">
        <span>ЛКМ: Вращение</span>
        <span>ПКМ: Панорама</span>
        <span>Колесо: Масштаб</span>
      </div>
    </div>
  );
}
