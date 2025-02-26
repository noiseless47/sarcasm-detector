"use client"

import { useState } from 'react';
import axios from 'axios';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import DarkModeToggle from '../../components/DarkModeToggle';
import ShareResults from '../../components/ShareResults';
import { motion } from 'framer-motion';

export default function SarcasmPage() {
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
    <div className="min-h-screen flex flex-col font-dmsans dark:bg-gray-900">
      <DarkModeToggle />
      <Header />
      
      <main className="flex-1 container mx-auto px-4 py-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-2xl mx-auto backdrop-blur-sm bg-white/80 dark:bg-gray-800/80 rounded-2xl shadow-xl p-8 border border-gray-100 dark:border-gray-700"
        >
          <div className="text-center mb-8">
            <h2 className="text-3xl font-londrina font-bold mb-4 dark:text-white">
              Detect Your Sentiment
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              Enter text to analyze your sentiment!
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <label className="block">
              <span className="text-gray-700 dark:text-gray-200 font-medium">
                Analyze Text
              </span>
              <textarea
                value={text}
                onChange={handleTextChange}
                placeholder="Enter text to analyze..."
                className="mt-2 block w-full rounded-lg border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white shadow-md focus:border-pink-500 focus:ring focus:ring-pink-200 focus:ring-opacity-50 p-4"
                rows={4}
              />
            </label>

            <motion.button
              type="submit"
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`w-full py-3 px-4 rounded-lg text-white font-medium ${
                loading
                  ? 'bg-gray-400'
                  : 'bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700'
              }`}
            >
              {loading ? 'Analyzing...' : 'Detect Sentiment'}
            </motion.button>
          </form>

          {error && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-4 p-4 bg-red-50 dark:bg-red-900 text-red-600 dark:text-red-200 rounded-lg"
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
              <h3 className="text-2xl font-londrina text-center mb-4 dark:text-white">
                {result.rating}
              </h3>
              <p className="text-gray-700 dark:text-gray-200 text-center italic">
                {result.fun_insult}
              </p>
              <ShareResults result={result} />
            </motion.div>
          )}
        </motion.div>
      </main>

      <Footer />
    </div>
  );
} 