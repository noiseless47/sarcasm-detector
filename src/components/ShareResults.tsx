'use client'

import { FiTwitter, FiFacebook, FiShare2 } from 'react-icons/fi'
import { motion } from 'framer-motion'

interface ShareResultsProps {
  result: {
    rating: string
    fun_insult: string
  }
}

export default function ShareResults({ result }: ShareResultsProps) {
  const shareText = `My sarcasm level: ${result.rating} - ${result.fun_insult}`
  
  const shareToTwitter = () => {
    window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}`, '_blank')
  }

  const shareToFacebook = () => {
    window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(window.location.href)}&quote=${encodeURIComponent(shareText)}`, '_blank')
  }

  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: 'Sarcasm Detector Result',
          text: shareText,
          url: window.location.href,
        })
      } else {
        // Fallback for browsers that don't support Web Share API
        alert('Share functionality not supported on this browser')
      }
    } catch (err) {
      console.error('Error sharing:', err)
    }
  }

  return (
    <div className="flex justify-center gap-6 mt-6">
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={shareToTwitter}
        className="p-3 rounded-full bg-gradient-to-r from-blue-400 to-blue-500 text-white hover:from-blue-500 hover:to-blue-600 shadow-lg hover:shadow-xl transition-all duration-300"
      >
        <FiTwitter className="w-6 h-6" />
      </motion.button>
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={shareToFacebook}
        className="p-3 rounded-full bg-gradient-to-r from-blue-600 to-blue-700 text-white hover:from-blue-700 hover:to-blue-800 shadow-lg hover:shadow-xl transition-all duration-300"
      >
        <FiFacebook className="w-6 h-6" />
      </motion.button>
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={handleShare}
        className="p-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-600 hover:to-pink-600 shadow-lg hover:shadow-xl transition-all duration-300"
      >
        <FiShare2 className="w-6 h-6" />
      </motion.button>
    </div>
  )
} 