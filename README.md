# 😏 Sarcasm Detector

A modern web application that analyzes text to detect sarcasm levels using AI. Built with Next.js, TypeScript, and TailwindCSS.

![Sarcasm Detector Demo](demo-screenshot.png)

## ✨ Features

- 🤖 AI-powered sarcasm detection
- 🌓 Dark/Light mode support
- 📱 Responsive design
- 🎨 Beautiful UI with smooth animations
- 🔄 Real-time analysis
- 📤 Easy sharing capabilities
- 🎯 Confidence score visualization

## 🚀 Tech Stack

- [Next.js 14](https://nextjs.org/) - React Framework
- [TypeScript](https://www.typescriptlang.org/) - Type Safety
- [TailwindCSS](https://tailwindcss.com/) - Styling
- [Framer Motion](https://www.framer.com/motion/) - Animations
- [Flask](https://flask.palletsprojects.com/) - Backend API

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sarcasm-detector.git
cd sarcasm-detector
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Set up environment variables:
```bash
cp .env.example .env.local
```
Edit `.env.local` with your configuration.

4. Run the development server:
```bash
npm run dev
# or
yarn dev
```

5. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## 📝 Usage

1. Enter text in the analysis box
2. Click "Detect Sarcasm"
3. View your sarcasm score and detailed analysis
4. Share results with friends

## 🔧 API Configuration

The application expects a Flask backend running on `http://127.0.0.1:5000`. Make sure to:

1. Set up the Flask backend (see backend repository)
2. Configure CORS settings
3. Ensure proper API response format:
```typescript
interface SarcasmResult {
  rating: string;
  fun_insult: string;
  explanation: string;
  confidence: number;
}
```

## 🎨 Customization

### Styling
The project uses TailwindCSS for styling. Customize the theme in `tailwind.config.js`:

```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        // Your custom colors
      }
    }
  }
}
```

### Fonts
We use custom fonts loaded through Next.js. Modify them in `app/layout.tsx`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Thanks to [Next.js](https://nextjs.org) for the amazing framework
- [Vercel](https://vercel.com) for hosting
- All contributors and supporters

## 📬 Contact

Your Name - [@yourusername](https://twitter.com/yourusername)

Project Link: [https://github.com/yourusername/sarcasm-detector](https://github.com/yourusername/sarcasm-detector)

---

Made with 😏 by [Your Name]
