import Navbar from "@/components/Navbar";
import VideoUpload from "@/components/VideoUpload";
import { Camera, Cpu, Box, Sparkles } from "lucide-react";

const features = [
  {
    icon: <Camera className="w-6 h-6 text-violet-400" />,
    title: "Загрузить видео",
    desc: "Запишите или загрузите видео вокруг объекта — подойдёт любая камера",
  },
  {
    icon: <Cpu className="w-6 h-6 text-violet-400" />,
    title: "GPU обработка",
    desc: "Реконструкция 3D модели на мощных GPU — только технология VlasovAI",
  },
  {
    icon: <Box className="w-6 h-6 text-violet-400" />,
    title: "Просмотр в 3D",
    desc: "Интерактивный 3D-просмотрщик — вращайте, масштабируйте, исследуйте",
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
            Только технология VlasovAI
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold mb-4 leading-tight">
            Превратите <span className="gradient-text">видео</span> в{" "}
            <span className="gradient-text">3D модель</span>
          </h1>
          <p className="text-lg text-zinc-400 max-w-xl mx-auto">
            Загрузите видео любой сцены или объекта. Наш конвейер
            создаст фотореалистичную 3D модель, которую можно
            исследовать прямо в браузере.
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
          <span>VlasovAI 4D map — Видео в 3D реконструкцию</span>
          <span>Только технология VlasovAI</span>
        </div>
      </footer>
    </div>
  );
}
