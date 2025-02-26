"use client";

import { motion } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Header() {
  const pathname = usePathname();

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Sarcasm', path: '/sarcasm' },
    { name: 'Sentiment', path: '/sentiment' },
    { name: 'Summarizer', path: '/summarizer' },
    { name: 'Language', path: '/language' },
  ];

  return (
    <header className="w-full bg-white dark:bg-gray-900 py-4 px-6 shadow-lg">
      <div className="max-w-7xl mx-auto">
        {/* Navigation */}
        <nav className="hidden lg:flex items-center justify-center gap-6">
          {navItems.map((item) => {
            const isActive = pathname === item.path;
            return (
              <Link 
                key={item.path} 
                href={item.path}
                className={`relative px-2 py-1 text-sm font-medium transition-colors
                  ${isActive 
                    ? 'text-pink-500 dark:text-pink-400' 
                    : 'text-gray-700 dark:text-gray-300 hover:text-pink-500 dark:hover:text-pink-400'
                  }`}
              >
                {item.name}
                {isActive && (
                  <motion.div
                    className="absolute -bottom-1 left-0 right-0 h-0.5 bg-pink-500 dark:bg-pink-400"
                    layoutId="activeNavIndicator"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Mobile Menu Button */}
        <button className="lg:hidden text-gray-700 dark:text-white">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    </header>
  );
} 