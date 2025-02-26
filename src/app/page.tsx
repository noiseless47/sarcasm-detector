"use client"

import { useState } from 'react';
import axios from 'axios';
import Header from '../components/Header';
import Footer from '../components/Footer';
import DarkModeToggle from '../components/DarkModeToggle';
import ShareResults from '../components/ShareResults';
import { motion } from 'framer-motion';

export default function Home() {
  const [text, setText] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://127.0.0.1:5000/', 
        { text },
        {
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
        }
      );
      setResult(response.data);
    } catch (error: any) {
      const errorMessage = error.response?.data?.details || error.message || 'An error occurred';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col font-dmsans dark:bg-gray-900 bg-gradient-to-b from-white to-pink-50 dark:from-gray-900 dark:to-gray-800">
      <DarkModeToggle />
      <Header />
      
      <main className="flex-1 container mx-auto px-4 py-12">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="max-w-2xl mx-auto backdrop-blur-sm bg-white/80 dark:bg-gray-800/80 rounded-2xl shadow-xl p-8 border border-gray-100 dark:border-gray-700"
        >
          <div className="text-center mb-10">
            <motion.h2 
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5 }}
              className="text-4xl font-londrina font-bold mb-4 dark:text-white bg-gradient-to-r from-pink-500 to-purple-600 bg-clip-text text-transparent"
            >
              Detect Your Sarcasm Level
            </motion.h2>
            <p className="text-gray-600 dark:text-gray-300 font-dmsans text-lg">
              Enter text to analyze your sarcasm score!
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-8">
            <label className="block">
              <span className="text-gray-700 dark:text-gray-200 font-medium text-lg mb-2 block">
                Analyze Text
              </span>
              <textarea
                value={text}
                onChange={handleTextChange}
                placeholder="Enter text to analyze..."
                className="mt-2 block w-full rounded-xl border-2 border-gray-200 dark:border-gray-600 dark:bg-gray-700 dark:text-white shadow-lg focus:border-pink-500 focus:ring-2 focus:ring-pink-200 focus:ring-opacity-50 p-4 transition-all duration-300 text-lg"
                rows={4}
              />
            </label>

            <motion.button
              type="submit"
              disabled={loading}
              whileHover={{ scale: 1.02, boxShadow: "0 5px 15px rgba(0,0,0,0.1)" }}
              whileTap={{ scale: 0.98 }}
              className={`w-full py-4 px-6 rounded-xl text-white font-medium text-lg ${
                loading
                  ? 'bg-gray-400'
                  : 'bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 shadow-lg hover:shadow-xl'
              } transition-all duration-300`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </span>
              ) : 'Detect Sarcasm'}
            </motion.button>
          </form>

          {error && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 p-4 bg-red-50 dark:bg-red-900/50 text-red-600 dark:text-red-200 rounded-xl font-dmsans border border-red-100 dark:border-red-800"
            >
              {error}
            </motion.div>
          )}

          {result && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 p-6 bg-gradient-to-r from-pink-50 to-purple-50 dark:from-gray-700 dark:to-gray-600 rounded-xl shadow-lg"
            >
              <h3 className="text-3xl font-londrina text-center mb-4 dark:text-white bg-gradient-to-r from-pink-500 to-purple-600 bg-clip-text text-transparent">
                {result.rating}
              </h3>
              <div className="space-y-4 mb-6">
                <p className="text-gray-700 dark:text-gray-200 font-dmsans text-center italic text-lg">
                  {result.fun_insult}
                </p>
                <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
                  <h4 className="font-londrina text-lg mb-2 text-gray-800 dark:text-gray-200">
                    Analysis Breakdown:
                  </h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    {result.explanation || "Your text shows signs of " + 
                    (result.rating.toLowerCase().includes("sincere") 
                      ? "sincerity and straightforwardness. The language used is direct and genuine, without typical markers of sarcasm like exaggeration or irony."
                      : "sarcastic undertones. The context and word choice suggest an ironic or playful meaning that differs from the literal interpretation."
                    )}
                  </p>
                  {result.confidence && (
                    <div className="mt-3">
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Confidence Score: {result.confidence}%
                      </p>
                      <div className="w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-full mt-1">
                        <div 
                          className="h-full bg-gradient-to-r from-pink-500 to-purple-600 rounded-full"
                          style={{ width: `${result.confidence}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <ShareResults result={result} />
            </motion.div>
          )}
        </motion.div>
      </main>

      <Footer />
    </div>
  );
}
