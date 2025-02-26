"use client";

import { TbBrandGithub, TbBrandLinkedin, TbBrandInstagram, TbBrandDiscord, TbBrandSpotify, TbBrandX } from 'react-icons/tb';
import { motion } from 'framer-motion';

export default function Footer() {
  return (
    <footer className="w-full py-8 px-8 bg-gradient-to-r from-gray-900 to-gray-800 text-white mt-12">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-6">
        <div className="flex gap-8">
          <motion.a 
            whileHover={{ scale: 1.2, y: -2 }}
            href="https://github.com/noiseless47/" 
            className="hover:text-pink-400 transition-colors"
          >
            <TbBrandGithub size={28} />
          </motion.a>
          <motion.a 
            whileHover={{ scale: 1.2, y: -2 }}
            href="https://www.x.com/AsishYeleti/" 
            className="hover:text-pink-400 transition-colors"
          >
            <TbBrandX size={28} />
          </motion.a>
          <motion.a 
            whileHover={{ scale: 1.2, y: -2 }}
            href="https://www.linkedin.com/in/asishkumaryeleti/" 
            className="hover:text-pink-400 transition-colors"
          >
            <TbBrandLinkedin size={28} />
          </motion.a>
          <motion.a 
            whileHover={{ scale: 1.2, y: -2 }}
            href="https://www.instagram.com/asish.k.y/" 
            className="hover:text-pink-400 transition-colors"
          >
            <TbBrandInstagram size={28} />
          </motion.a>
          <motion.a 
            whileHover={{ scale: 1.2, y: -2 }}
            href="https://discord.gg/your-discord-invite" 
            className="hover:text-pink-400 transition-colors"
          >
            <TbBrandDiscord size={28} />
          </motion.a>
          <motion.a 
            whileHover={{ scale: 1.2, y: -2 }}
            href="https://open.spotify.com/user/o1mhl5cn2icv8bc2z81fgaiqc" 
            className="hover:text-pink-400 transition-colors"
          >
            <TbBrandSpotify size={28} />
          </motion.a>
        </div>
        <div className="h-px w-32 bg-gradient-to-r from-transparent via-pink-400 to-transparent"></div>
        <p className="text-gray-400 text-sm font-medium">
          © 2024 Sarcasm Detector. Made with <span className="text-pink-400">😏</span> by Your Name
        </p>
      </div>
    </footer>
  );
} 