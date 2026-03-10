"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Upload,
  Video,
  X,
  Zap,
  Settings2,
  Sparkles,
  ArrowRight,
  Loader2,
} from "lucide-react";

type Quality = "fast" | "balanced" | "ultra";
type MeshMethod = "poisson" | "ball_pivoting" | "alpha";
type ExportFormat = "splat" | "obj" | "dae" | "stl" | "glb" | "ply_mesh";

const qualityOptions: {
  value: Quality;
  label: string;
  desc: string;
  icon: React.ReactNode;
  frames: number;
  density: number;
}[] = [
  {
    value: "fast",
    label: "Быстрый",
    desc: "~50 кадров, базовая детализация",
    icon: <Zap className="w-4 h-4" />,
    frames: 50,
    density: 1.0,
  },
  {
    value: "balanced",
    label: "Баланс",
    desc: "~100 кадров, хорошая детализация",
    icon: <Settings2 className="w-4 h-4" />,
    frames: 100,
    density: 2.0,
  },
  {
    value: "ultra",
    label: "Ультра",
    desc: "~150 кадров, максимальная детализация",
    icon: <Sparkles className="w-4 h-4" />,
    frames: 150,
    density: 3.0,
  },
];

export default function VideoUpload() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [quality, setQuality] = useState<Quality>("balanced");
  const [generateMesh, setGenerateMesh] = useState(false);
  const [meshMethod, setMeshMethod] = useState<MeshMethod>("poisson");
  const [exportFormats, setExportFormats] = useState<ExportFormat[]>(["splat"]);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>("");
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [hydrated, setHydrated] = useState(false);
  useEffect(() => {
    console.log("[VideoUpload] React hydrated successfully");
    setHydrated(true);
    // Restore active job from sessionStorage (survives HMR reloads)
    const saved = sessionStorage.getItem("4d_active_job");
    if (saved) {
      console.log("[VideoUpload] Restoring job from sessionStorage:", saved);
      setJobId(saved);
      setUploading(true);
      setStatusMsg("Переподключение к задаче...");
    }
  }, []);

  // Poll job status inline (no page navigation needed)
  useEffect(() => {
    if (!jobId) return;
    let active = true;
    const poll = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!active) return;
        setJobStatus(data.status);
        setStatusMsg(data.message || "Обработка...");
        setUploadProgress(data.progress || 0);
        if (data.status === "completed" && data.modelUrl) {
          setModelUrl(data.modelUrl);
          sessionStorage.removeItem("4d_active_job");
          if (pollRef.current) clearInterval(pollRef.current);
        }
        if (data.status === "failed") {
          setError(data.message || "Обработка не удалась");
          setUploading(false);
          sessionStorage.removeItem("4d_active_job");
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (e) {
        console.log("[poll] fetch error, retrying...", e);
      }
    };
    poll();
    pollRef.current = setInterval(poll, 3000);
    return () => { active = false; if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobId]);

  const handleFile = useCallback((f: File) => {
    setError(null);
    const allowedTypes = [
      "video/mp4",
      "video/webm",
      "video/quicktime",
      "video/x-msvideo",
    ];
    if (!allowedTypes.includes(f.type)) {
      setError("Неподдерживаемый формат. Используйте MP4, WebM, MOV или AVI.");
      return;
    }
    if (f.size > 500 * 1024 * 1024) {
      setError("Файл слишком большой. Максимум 500МБ.");
      return;
    }
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const removeFile = useCallback(() => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [preview]);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setUploadProgress(0);
    setStatusMsg("Загрузка видео...");

    try {
      const selectedQuality = qualityOptions.find((q) => q.value === quality);
      const maxFrames = selectedQuality?.frames || 200;
      const densityFactor = selectedQuality?.density || 2.0;

      const formData = new FormData();
      formData.append("video", file);
      formData.append("quality", quality);
      formData.append("maxFrames", String(maxFrames));
      formData.append("density_factor", String(densityFactor));
      formData.append("generate_mesh", String(generateMesh));
      formData.append("mesh_method", meshMethod);
      formData.append("export_formats", JSON.stringify(exportFormats));

      // Simulate progress (localhost upload is instant)
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 15, 80));
      }, 300);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(90);

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Ошибка загрузки");
      }

      const data = await res.json();
      setUploadProgress(0);
      setStatusMsg("Сжатие и отправка на GPU...");
      sessionStorage.setItem("4d_active_job", data.jobId);
      setJobId(data.jobId); // triggers polling useEffect
    } catch (err) {
      console.error("[upload] Error:", err);
      setError(err instanceof Error ? err.message : "Ошибка загрузки");
      setUploading(false);
      setUploadProgress(0);
      setStatusMsg("");
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* Drop Zone */}
      {!file ? (
        <div
          className={`upload-zone rounded-2xl p-12 text-center transition-all ${
            dragOver ? "drag-over" : ""
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <Upload className="w-12 h-12 text-zinc-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-zinc-200 mb-2">
            Перетащите видео сюда
          </h3>
          <p className="text-sm text-zinc-500 mb-4">
            или нажмите кнопку ниже — MP4, WebM, MOV, AVI до 500МБ
          </p>
          {/* Visible file input styled as a button — works without JS */}
          <label
            htmlFor="video-file-input"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-semibold cursor-pointer transition-colors"
          >
            Выбрать видеофайл
            <input
              id="video-file-input"
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
              className="sr-only"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
          </label>
          <p className="text-xs text-zinc-600 mt-4">
            Для лучшего результата медленно обойдите объект/сцену плавным
            кругом
          </p>
        </div>
      ) : (
        /* Video Preview */
        <div className="rounded-2xl overflow-hidden bg-zinc-900 border border-zinc-800">
          <div className="relative">
            <video
              src={preview || undefined}
              className="w-full max-h-80 object-contain bg-black"
              controls
              muted
            />
            {!uploading && (
              <button
                onClick={removeFile}
                className="absolute top-3 right-3 p-2 rounded-full bg-black/60 hover:bg-black/80 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          <div className="p-4 flex items-center gap-3">
            <Video className="w-5 h-5 text-violet-400 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{file.name}</p>
              <p className="text-xs text-zinc-500">
                {(file.size / (1024 * 1024)).toFixed(1)} MB
              </p>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Quality Selection */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-zinc-400">
          Качество реконструкции
        </h4>
        <div className="grid grid-cols-3 gap-3">
          {qualityOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setQuality(opt.value)}
              disabled={uploading}
              className={`p-4 rounded-xl text-left transition-all ${
                quality === opt.value
                  ? "bg-violet-600/20 border-violet-500/50 border"
                  : "bg-zinc-900 border border-zinc-800 hover:border-zinc-700"
              } ${uploading ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={
                    quality === opt.value ? "text-violet-400" : "text-zinc-500"
                  }
                >
                  {opt.icon}
                </span>
                <span className="text-sm font-semibold">{opt.label}</span>
              </div>
              <p className="text-xs text-zinc-500">{opt.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Mesh Generation Options */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-zinc-400">
          3D генерация и форматы
        </h4>
        
        {/* Generate Mesh Toggle */}
        <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/50">
          <input
            type="checkbox"
            id="generate-mesh"
            checked={generateMesh}
            onChange={(e) => setGenerateMesh(e.target.checked)}
            disabled={uploading}
            className="w-4 h-4 text-violet-600 bg-zinc-800 border-zinc-600 rounded focus:ring-violet-500"
          />
          <label htmlFor="generate-mesh" className="text-sm text-zinc-300 cursor-pointer">
            Создать 3D mesh (полигональную модель)
          </label>
        </div>

        {/* Mesh Method Selection */}
        {generateMesh && (
          <div className="space-y-2">
            <h5 className="text-xs font-medium text-zinc-500">Метод генерации mesh:</h5>
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: "poisson", label: "Poisson", desc: "Качественный" },
                { value: "ball_pivoting", label: "Ball Pivot", desc: "Быстрый" },
                { value: "alpha", label: "Alpha Shape", desc: "Простой" },
              ].map((method) => (
                <button
                  key={method.value}
                  onClick={() => setMeshMethod(method.value as MeshMethod)}
                  disabled={uploading}
                  className={`p-2 rounded-lg text-xs transition-all ${
                    meshMethod === method.value
                      ? "bg-violet-600/20 border-violet-500/50 border"
                      : "bg-zinc-900 border border-zinc-800 hover:border-zinc-700"
                  } ${uploading ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <div className="font-semibold">{method.label}</div>
                  <div className="text-zinc-500">{method.desc}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Export Format Selection */}
        <div className="space-y-2">
          <h5 className="text-xs font-medium text-zinc-500">Форматы экспорта:</h5>
          <div className="grid grid-cols-3 gap-2">
            {[
              { value: "splat", label: "Splat", desc: "Для веба", always: true },
              { value: "obj", label: "OBJ", desc: "3D модели" },
              { value: "glb", label: "GLB", desc: "WebGL" },
              { value: "dae", label: "DAE", desc: "Collada" },
              { value: "stl", label: "STL", desc: "3D печать" },
              { value: "ply_mesh", label: "PLY", desc: "Point cloud" },
            ].map((format) => (
              <button
                key={format.value}
                onClick={() => {
                  if (format.always) return;
                  setExportFormats(prev =>
                    prev.includes(format.value as ExportFormat)
                      ? prev.filter(f => f !== format.value)
                      : [...prev, format.value as ExportFormat]
                  );
                }}
                disabled={uploading || format.always}
                className={`p-2 rounded-lg text-xs transition-all ${
                  exportFormats.includes(format.value as ExportFormat) || format.always
                    ? "bg-violet-600/20 border-violet-500/50 border"
                    : "bg-zinc-900 border border-zinc-800 hover:border-zinc-700"
                } ${uploading || format.always ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                <div className="font-semibold">{format.label}</div>
                <div className="text-zinc-500">{format.desc}</div>
                {format.always && <div className="text-violet-400 text-xs">Всегда</div>}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Upload Progress */}
      {uploading && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-400">{statusMsg || "Обработка..."}</span>
            <span className="text-violet-400">{uploadProgress}%</span>
          </div>
          <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-600 to-violet-400 rounded-full transition-all duration-300 progress-pulse"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* View 3D Model (when complete) */}
      {modelUrl && (
        <a
          href={`/viewer/?id=${jobId}`}
          className="w-full py-4 px-6 rounded-xl font-semibold text-base flex items-center justify-center gap-3 transition-all bg-gradient-to-r from-green-600 to-emerald-500 hover:from-green-500 hover:to-emerald-400 text-white shadow-lg shadow-green-600/20"
        >
          Просмотр 3D модели
          <ArrowRight className="w-5 h-5" />
        </a>
      )}

      {/* Submit Button */}
      {!jobId && (
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className={`w-full py-4 px-6 rounded-xl font-semibold text-base flex items-center justify-center gap-3 transition-all ${
            !file || uploading
              ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
              : "bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white shadow-lg shadow-violet-600/20"
          }`}
        >
          {uploading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Обработка...
            </>
          ) : (
            <>
              Создать 3D модель
              <ArrowRight className="w-5 h-5" />
            </>
          )}
        </button>
      )}
    </div>
  );
}
