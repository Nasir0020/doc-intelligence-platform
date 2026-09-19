import type { Metadata } from "next";
import "./globals.css";

// This is a SERVER component (no "use client" — it doesn't need
// interactivity, only to define shared page structure/metadata).
// Next.js's App Router wraps every page with this layout automatically
// — it's the one place <html>/<body> tags belong in the entire app.
export const metadata: Metadata = {
  title: "Document Intelligence Platform",
  description: "Ask questions about your documents, with citations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-white text-gray-900">{children}</body>
    </html>
  );
}
