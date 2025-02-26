'use client'

import { FiSun, FiMoon } from 'react-icons/fi'
import { useDarkMode } from '../context/DarkModeContext'

export default function DarkModeToggle() {
  const { darkMode, toggleDarkMode } = useDarkMode()

  return (
    <button
      onClick={toggleDarkMode}
      className="fixed top-4 right-4 p-3 rounded-full bg-gray-200 dark:bg-gray-800 transition-all duration-300 hover:scale-110 z-50"
      aria-label="Toggle dark mode"
    >
      {darkMode ? (
        <FiSun className="w-6 h-6 text-yellow-500" />
      ) : (
        <FiMoon className="w-6 h-6 text-gray-700" />
      )}
    </button>
  )
} 