"use client";

export default function Header() {
  return (
    <header className="w-full py-4 px-8 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-4xl">😏</span>
          <h1 className="text-2xl font-bold text-white">Sarcasm Detector</h1>
        </div>
        <nav>
          <ul className="flex gap-6 text-white">
            <li>
              <a href="/" className="hover:text-pink-200 transition-colors">
                Home
              </a>
            </li>
            <li>
              <a href="/about" className="hover:text-pink-200 transition-colors">
                About
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
} 