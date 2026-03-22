import { useState, useEffect } from 'react';
import { motion } from 'motion/react';

interface HeroProps {
  onStart: () => void;
}

const STICKER_ASSETS = [
  {
    id: 'googly-eyes',
    src: '/googlyeyes.png',
    alt: 'Googly Eyes',
    baseClass: 'w-24 md:w-32'
  },
  {
    id: 'cat-stamp',
    src: '/catstamp.jpg',
    alt: 'Cat Stamp',
    baseClass: 'stamp-effect w-28 md:w-40'
  },
  {
    id: 'fish-stamp',
    src: '/fishstamp.jpg',
    alt: 'Fish Stamp',
    baseClass: 'stamp-effect w-28 md:w-40'
  },
  {
    id: 'seal-stamp',
    src: '/stealstampfr.png',
    alt: 'Seal Stamp',
    baseClass: 'w-28 md:w-40'
  },
  {
    id: 'safety-pin',
    src: '/safety pin.png',
    alt: 'Safety Pin',
    baseClass: 'w-20 md:w-28'
  },
  {
    id: 'starr',
    src: '/starr.png',
    alt: 'Star',
    baseClass: 'w-24 md:w-32'
  }
];

export default function Hero({ onStart }: HeroProps) {
  const [placedStickers, setPlacedStickers] = useState<any[]>([]);

  useEffect(() => {
    // Randomize positions on mount so they are different every time the user opens the site
    // Define 4 distinct corner zones to completely avoid the center text/buttons
    // Pushed further to the edges to avoid the larger center image
    const zones = [
      { left: [2, 12], top: [2, 15] },   // Top Left
      { left: [75, 88], top: [2, 15] },  // Top Right
      { left: [2, 12], top: [75, 88] },  // Bottom Left
      { left: [75, 88], top: [75, 88] }, // Bottom Right
      { left: [2, 12], top: [40, 55] },  // Middle Left
      { left: [75, 88], top: [40, 55] }  // Middle Right
    ];
    
    // Shuffle zones so stickers appear in different corners each time
    const shuffledZones = [...zones].sort(() => Math.random() - 0.5);

    const randomized = STICKER_ASSETS.map((sticker, index) => {
      const zone = shuffledZones[index % shuffledZones.length];
      
      const leftPos = Math.floor(Math.random() * (zone.left[1] - zone.left[0])) + zone.left[0] + '%';
      const topPos = Math.floor(Math.random() * (zone.top[1] - zone.top[0])) + zone.top[0] + '%';
      
      const rotate = Math.floor(Math.random() * 60) - 30; // -30 to 30 degrees
      
      return {
        ...sticker,
        left: leftPos,
        top: topPos,
        rotate: rotate,
        delay: 0.8 + Math.random() * 1.0
      };
    });
    setPlacedStickers(randomized);
  }, []);

  return (
    <div className="h-screen w-full flex flex-col items-center justify-center relative overflow-hidden">
      {/* Grid Lines are global in body, so we just ensure they aren't masked too much */}
      
      {/* Randomly Placed Stickers */}
      {placedStickers.map((sticker) => (
        <motion.img
          key={sticker.id}
          initial={{ opacity: 0, scale: 0.5, rotate: 0 }}
          animate={{ opacity: 1, scale: 1, rotate: sticker.rotate }}
          transition={{ 
            delay: sticker.delay, 
            duration: 0.8, 
            type: "spring", 
            stiffness: 100 
          }}
          whileHover={{ scale: 1.1, rotate: 0, zIndex: 50 }}
          src={sticker.src}
          alt={sticker.alt}
          className={`sticker-base !absolute z-0 hover:z-10 ${sticker.baseClass}`}
          style={{ left: sticker.left, top: sticker.top }}
          referrerPolicy="no-referrer"
        />
      ))}

      <div className="relative z-20 text-center space-y-12 px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9, rotate: -2 }}
          animate={{ opacity: 1, scale: 1, rotate: 0 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          className="relative inline-flex items-center justify-center py-20 px-32 md:py-24 md:px-40"
        >
          <img 
            src="/rippaperr.png" 
            alt="Ripped Paper Background" 
            className="absolute inset-0 w-full h-full object-contain scale-[1.5] md:scale-[1.6] rotate-[-8deg] -z-10 drop-shadow-2xl pointer-events-none"
          />
          <h1 className="text-7xl md:text-9xl font-bold tracking-tighter text-envision-pink font-ubuntu relative z-10">
            envision
          </h1>
        </motion.div>

        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 1 }}
          className="text-xl md:text-2xl text-royal-blue font-semibold tracking-[0.5em] uppercase"
        >
          virtually designing you
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 0.8 }}
          className="flex justify-center w-full"
        >
          <button 
            onClick={onStart}
            className="washi-tape-button font-bold"
          >
            get dressed
          </button>
        </motion.div>
      </div>
    </div>
  );
}
