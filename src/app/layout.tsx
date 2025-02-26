"use client";

import './globals.css'
import { DM_Sans, Londrina_Shadow } from 'next/font/google'
import { ThemeProvider } from 'next-themes'

const dmsans = DM_Sans({
  subsets: ['latin'],
  weight: ['400', '600'],
  variable: '--font-dmsans',
})

const londrinashadow = Londrina_Shadow({
  subsets: ['latin'],
  weight: ['400'],
  variable: '--font-londrina',
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <title>Sarcasm Detector</title>
      </head>
      <body className={`${dmsans.variable} ${londrinashadow.variable}`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
