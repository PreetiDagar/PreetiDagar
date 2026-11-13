import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "GRIDMIND AI | Energy Intelligence", description: "AI-powered energy forecasting, anomaly intelligence and digital twin simulation." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
