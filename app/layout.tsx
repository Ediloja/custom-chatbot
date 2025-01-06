import type { Metadata } from "next";
import { Inter } from "next/font/google";
import cx from "./lib/cx";
import "./globals.css";

const inter = Inter({
    subsets: ["latin"],
});

export const metadata: Metadata = {
    title: "RAG Chatbot",
    description: "Custom RAG Chatbot for University Courses at UTPL",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className="scroll-smooth antialiased">
            <body
                className={cx(inter.className, "bg-white text-sm md:text-base")}
            >
                {children}
            </body>
        </html>
    );
}
