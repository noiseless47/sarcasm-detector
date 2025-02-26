'use client'

import { createContext, useContext, useEffect, useState } from 'react'

type DarkModeContextType = {
  darkMode: boolean
  toggleDarkMode: () => void
}

const DarkModeContext = createContext<DarkModeContextType>({
  darkMode: false,
  toggleDarkMode: () => {},
})

export function DarkModeProvider({ children }: { children: React.ReactNode }) {
  const [darkMode, setDarkMode] = useState(false)

  useEffect(() => {
    // Check local storage and system preference on mount
    const savedMode = localStorage.getItem('darkMode')
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches

    setDarkMode(savedMode ? JSON.parse(savedMode) : systemPrefersDark)
  }, [])

  useEffect(() => {
    // Update class and save preference when darkMode changes
    if (darkMode) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('darkMode', 'true')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('darkMode', 'false')
    }
  }, [darkMode])

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
  }

  return (
    <DarkModeContext.Provider value={{ darkMode, toggleDarkMode }}>
      {children}
    </DarkModeContext.Provider>
  )
}

export const useDarkMode = () => useContext(DarkModeContext) 