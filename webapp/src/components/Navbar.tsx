"use client";

import Link from "next/link";
import { Box, Layers3 } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-card">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="relative">
              <Box className="w-8 h-8 text-violet-500 group-hover:text-violet-400 transition-colors" />
              <Layers3 className="w-4 h-4 text-violet-300 absolute -bottom-1 -right-1" />
            </div>
            <span className="text-xl font-bold gradient-text">4Demension</span>
          </Link>

          <div className="flex items-center gap-6">
            <Link
              href="/"
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              Upload
            </Link>
            <a
              href="https://github.com/ByteDance-Seed/depth-anything-3"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              Depth Anything V3
            </a>
            <a
              href="https://www.runpod.io"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              RunPod
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
}
