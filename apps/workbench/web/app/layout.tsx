import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import { workbenchVisualTokens } from "@/lib/visual-tokens";
import "./globals.css";

const sans = localFont({
  src: "./fonts/plus-jakarta-sans.woff2",
  weight: "400 800",
  variable: "--font-sans",
});

const mono = localFont({
  src: "./fonts/jetbrains-mono.woff2",
  weight: "400 700",
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Emperor Model Workbench",
  description:
    "Inspect Emperor model packages, review Runtime Defaults, explore model graphs, analyze logs and monitors, and plan or observe training.",
};

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: workbenchVisualTokens.bg,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
