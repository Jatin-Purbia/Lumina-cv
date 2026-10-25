import type { Metadata, Viewport } from "next";
import type { ReactElement, ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lumina CV",
  description: "Enhance low-light photos and compare them with the original.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  colorScheme: "light dark",
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>): ReactElement {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
