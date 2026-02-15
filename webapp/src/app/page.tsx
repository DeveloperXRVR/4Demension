import Navbar from "@/components/Navbar";
import VideoUpload from "@/components/VideoUpload";
import { Camera, Cpu, Box, Sparkles } from "lucide-react";

const features = [
  {
    icon: <Camera className="w-6 h-6 text-violet-400" />,
    title: "Upload Video",
    desc: "Record or upload video walking around your subject — any camera works",
  },
  {
    icon: <Cpu className="w-6 h-6 text-violet-400" />,
    title: "GPU Processing",
    desc: "Depth Anything V3 + 3D Gaussian Splatting on RunPod GPU cluster",
  },
  {
    icon: <Box className="w-6 h-6 text-violet-400" />,
    title: "View in 3D",
    desc: "Interactive Gaussian Splat viewer — rotate, zoom, and explore your model",
  },
];

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 pt-24 pb-16 px-4">
        {/* Hero */}
        <div className="max-w-3xl mx-auto text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-violet-600/10 border border-violet-500/20 text-violet-400 text-xs font-medium mb-6">
            <Sparkles className="w-3.5 h-3.5" />
            Powered by Depth Anything V3 & 3D Gaussian Splatting
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold mb-4 leading-tight">
            Turn <span className="gradient-text">Video</span> into{" "}
            <span className="gradient-text">3D Models</span>
          </h1>
          <p className="text-lg text-zinc-400 max-w-xl mx-auto">
            Upload video footage of any scene or object. Our pipeline
            reconstructs a photorealistic 3D Gaussian Splat model you can
            explore right in your browser.
          </p>
        </div>

        {/* Features */}
        <div className="max-w-2xl mx-auto grid grid-cols-3 gap-4 mb-12">
          {features.map((f, i) => (
            <div
              key={i}
              className="p-4 rounded-xl bg-zinc-900/50 border border-zinc-800/50 text-center"
            >
              <div className="flex justify-center mb-3">{f.icon}</div>
              <h3 className="text-sm font-semibold mb-1">{f.title}</h3>
              <p className="text-xs text-zinc-500">{f.desc}</p>
            </div>
          ))}
        </div>

        {/* Upload Component */}
        <VideoUpload />
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/50 py-6 px-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-xs text-zinc-600">
          <span>4Demension — Video to 3D Reconstruction</span>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/ByteDance-Seed/depth-anything-3"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-400 transition-colors"
            >
              Depth Anything V3
            </a>
            <a
              href="https://www.runpod.io"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-400 transition-colors"
            >
              RunPod
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
