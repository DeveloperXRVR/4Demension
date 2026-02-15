"use client";

import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
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

const qualityOptions: {
  value: Quality;
  label: string;
  desc: string;
  icon: React.ReactNode;
  frames: number;
}[] = [
  {
    value: "fast",
    label: "Fast",
    desc: "~100 frames, quick preview quality",
    icon: <Zap className="w-4 h-4" />,
    frames: 100,
  },
  {
    value: "balanced",
    label: "Balanced",
    desc: "~200 frames, good detail & speed",
    icon: <Settings2 className="w-4 h-4" />,
    frames: 200,
  },
  {
    value: "ultra",
    label: "Ultra",
    desc: "~400 frames, maximum detail",
    icon: <Sparkles className="w-4 h-4" />,
    frames: 400,
  },
];

export default function VideoUpload() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [quality, setQuality] = useState<Quality>("balanced");
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback((f: File) => {
    setError(null);
    const allowedTypes = [
      "video/mp4",
      "video/webm",
      "video/quicktime",
      "video/x-msvideo",
    ];
    if (!allowedTypes.includes(f.type)) {
      setError("Unsupported format. Use MP4, WebM, MOV, or AVI.");
      return;
    }
    if (f.size > 500 * 1024 * 1024) {
      setError("File too large. Maximum 500MB.");
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

    try {
      const formData = new FormData();
      formData.append("video", file);
      formData.append("quality", quality);
      const selectedQuality = qualityOptions.find((q) => q.value === quality);
      formData.append("maxFrames", String(selectedQuality?.frames || 200));

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Upload failed");
      }

      const data = await res.json();
      router.push(`/processing/${data.jobId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* Drop Zone */}
      {!file ? (
        <div
          className={`upload-zone rounded-2xl p-12 text-center cursor-pointer transition-all ${
            dragOver ? "drag-over" : ""
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />
          <Upload className="w-12 h-12 text-zinc-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-zinc-200 mb-2">
            Drop your video here
          </h3>
          <p className="text-sm text-zinc-500 mb-4">
            or click to browse — MP4, WebM, MOV, AVI up to 500MB
          </p>
          <p className="text-xs text-zinc-600">
            For best results, walk slowly around the object/scene in a smooth
            circle
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
          Reconstruction Quality
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

      {/* Upload Progress */}
      {uploading && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-400">Uploading video...</span>
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

      {/* Submit Button */}
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
            Processing...
          </>
        ) : (
          <>
            Reconstruct 3D Model
            <ArrowRight className="w-5 h-5" />
          </>
        )}
      </button>
    </div>
  );
}
