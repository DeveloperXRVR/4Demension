import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VlasovAI 4D map — Видео в 3D модель",
  description: "Загрузите видео и получите фотореалистичную 3D модель. Только технология VlasovAI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body className="antialiased" style={{ fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
        {children}
      </body>
    </html>
  );
}
