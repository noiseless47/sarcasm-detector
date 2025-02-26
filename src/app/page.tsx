"use client"

import Header from '../components/Header';
import Footer from '../components/Footer';
import DarkModeToggle from '../components/DarkModeToggle';
import { motion } from 'framer-motion';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col font-dmsans dark:bg-gray-900">
      <DarkModeToggle />
      <Header />
      
      <main className="flex-1 container mx-auto px-4 py-12">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="max-w-4xl mx-auto text-center"
        >
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-pink-500 to-purple-600 bg-clip-text text-transparent">
            Your Personal Text Analyzer
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-12">
            Analyze your text with advanced AI algorithms. Choose from multiple analysis tools including sarcasm detection, sentiment analysis, text summarization, and language detection.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                title: "Sarcasm Detection",
                description: "Detect subtle hints of sarcasm in any text",
                link: "/sarcasm"
              },
              {
                title: "Sentiment Analysis",
                description: "Understand the emotional tone of your text",
                link: "/sentiment"
              },
              {
                title: "Text Summarization",
                description: "Get concise summaries of long texts",
                link: "/summarizer"
              }
              // Add more cards as needed
            ].map((card, index) => (
              <motion.a
                key={index}
                href={card.link}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-xl hover:shadow-2xl transition-all duration-300"
              >
                <h3 className="text-xl font-bold mb-2 text-gray-800 dark:text-white">
                  {card.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300">
                  {card.description}
                </p>
              </motion.a>
            ))}
          </div>
        </motion.div>
      </main>

      <Footer />
    </div>
  );
}
