"use client"

import { useTheme } from "next-themes"
import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { FiSun, FiMoon } from "react-icons/fi"

export default function DarkModeToggle() {
  const [mounted, setMounted] = useState(false)
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="fixed top-4 right-4 z-50">
      <button
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="relative w-16 h-8 rounded-full bg-gray-200 dark:bg-gray-700 transition-colors duration-300"
      >
        <motion.div
          className="absolute top-1 left-1 w-6 h-6 rounded-full flex items-center justify-center"
          animate={{
            x: theme === "dark" ? 32 : 0,
            backgroundColor: theme === "dark" ? "#6366f1" : "#f59e0b",
          }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
        >
          {theme === "dark" ? (
            <FiMoon className="text-white w-4 h-4" />
          ) : (
            <FiSun className="text-white w-4 h-4" />
          )}
        </motion.div>
      </button>
    </div>
  )
} 