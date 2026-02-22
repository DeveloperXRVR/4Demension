"use client";

import { useEffect, useRef, useState } from "react";
import {
  Loader2,
  Maximize2,
  Minimize2,
  RotateCcw,
  Download,
  Info,
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
  const wrapperRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [pointCount, setPointCount] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    let disposed = false;

    const init = async () => {
      try {
        const GaussianSplats3D = await import(
          "@mkkellogg/gaussian-splats-3d"
        );

        if (disposed) return;

        const viewer = new GaussianSplats3D.Viewer({
          rootElement: container,
          cameraUp: [0, 1, 0],
          initialCameraPosition: [0, 2, 5],
          initialCameraLookAt: [0, 0, 0],
          gpuAcceleratedSort: true,
          sharedMemoryForWorkers: false,
          renderMode: GaussianSplats3D.RenderMode.OnChange,
          sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
          logLevel: GaussianSplats3D.LogLevel.None,
          sphericalHarmonicsDegree: 0,
          antialiased: false,
          focalAdjustment: 1.0,
        });

        viewerRef.current = viewer;

        await viewer.addSplatScene(modelUrl, {
          splatAlphaRemovalThreshold: 5,
          showLoadingUI: false,
          progressiveLoad: true,
        });

        if (disposed) {
          try { viewer.dispose(); } catch {}
          return;
        }

        viewer.start();

        // Try to get splat count
        try {
          const count = viewer.getSplatCount?.() ?? 0;
          setPointCount(count);
        } catch {
          // getSplatCount may not exist in all versions
        }

        setLoading(false);
      } catch (err) {
        if (!disposed) {
          console.error("GS3D viewer error:", err);
          setLoadError(
            err instanceof Error ? err.message : "Failed to load 3D model"
          );
          setLoading(false);
        }
      }
    };

    init();

    return () => {
      disposed = true;
      if (viewerRef.current) {
        try {
          viewerRef.current.stop?.();
          viewerRef.current.dispose?.();
        } catch (e) {
          console.warn("Viewer cleanup:", e);
        }
        viewerRef.current = null;
      }
      // Remove any leftover canvases
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelUrl]);

  const resetCamera = () => {
    // Dispose and re-init would be heavy; just reload
    if (viewerRef.current) {
      try {
        const cam = viewerRef.current.camera;
        if (cam) {
          cam.position.set(0, 2, 5);
          cam.lookAt(0, 0, 0);
        }
      } catch {}
    }
  };

  const toggleFullscreen = () => {
    const el = wrapperRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = modelUrl;
    a.download = `4demension-${jobId.slice(0, 8)}.splat`;
    a.click();
  };

  return (
    <div ref={wrapperRef} className="w-full h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/90 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-violet-400" />
          <span className="text-sm font-medium">3D Gaussian Splat Viewer</span>
          {pointCount > 0 && (
            <span className="text-xs text-zinc-500 ml-2">
              {pointCount.toLocaleString()} splats
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
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
      <div
        ref={containerRef}
        className="flex-1 relative min-h-0 splat-viewer"
        style={{ background: "#111" }}
      >
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/80 z-10 pointer-events-none">
            <Loader2 className="w-10 h-10 animate-spin text-violet-400 mb-4" />
            <p className="text-sm text-zinc-400">Loading 3D model...</p>
          </div>
        )}
        {loadError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/80 z-10">
            <Info className="w-10 h-10 text-red-400 mb-4" />
            <p className="text-sm text-red-400 mb-2">Failed to load model</p>
            <p className="text-xs text-zinc-500 max-w-md text-center">
              {loadError}
            </p>
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
