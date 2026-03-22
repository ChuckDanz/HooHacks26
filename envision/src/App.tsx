import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import Hero from './components/Hero';
import Preferences from './components/Preferences';
import TryOn from './components/TryOn';
import { getOrCreateSession } from './utils/session';

export type FlowStage = 'hero' | 'preferences' | 'tryon';

export default function App() {
  const [stage, setStage] = useState<FlowStage>('hero');
  const [sessionId] = useState<string>(() => getOrCreateSession());
  const [userData, setUserData] = useState({
    gender: '',
    topSize: '',
    bottomSize: '',
    measurements: null,
    modelImage: null as string | null,
  });

  const nextStage = (next: FlowStage) => setStage(next);

  return (
    <div className="min-h-screen w-full relative overflow-hidden">
      <div className="fixed top-0 left-0 w-full h-16 bg-white/90 backdrop-blur-md z-50 flex items-center justify-center border-b border-royal-blue/20">
        <div className="flex gap-12 text-2xl font-covered tracking-widest uppercase">
          <button onClick={() => setStage('hero')} className={`transition-all ${stage === 'hero' ? 'text-envision-pink' : 'text-royal-blue/30 hover:text-royal-blue'}`}>START</button>
          <button onClick={() => setStage('tryon')} className={`transition-all ${stage === 'tryon' ? 'text-envision-pink' : 'text-royal-blue/30 hover:text-royal-blue'}`}>CLOSET</button>
          <button onClick={() => setStage('preferences')} className={`transition-all ${stage === 'preferences' ? 'text-envision-pink' : 'text-royal-blue/30 hover:text-royal-blue'}`}>PREFERENCES</button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {stage === 'hero' && (
          <motion.div key="hero" initial={{ y: 0 }} exit={{ y: '100%' }} transition={{ type: 'spring', damping: 25, stiffness: 100 }} className="absolute inset-0 z-40">
            <Hero onStart={() => nextStage('tryon')} />
          </motion.div>
        )}

        {stage === 'preferences' && (
          <motion.div key="preferences" initial={{ y: '-100%' }} animate={{ y: 0 }} exit={{ y: '100%' }} transition={{ type: 'spring', damping: 25, stiffness: 100 }} className="absolute inset-0 z-30">
            <Preferences
              sessionId={sessionId}
              onComplete={(data) => {
                setUserData(prev => ({ ...prev, ...data }));
                nextStage('tryon');
              }}
            />
          </motion.div>
        )}

        {stage === 'tryon' && (
          <motion.div key="tryon" initial={{ y: '-100%' }} animate={{ y: 0 }} transition={{ type: 'spring', damping: 25, stiffness: 100 }} className="absolute inset-0 z-20">
            <TryOn userData={userData} sessionId={sessionId} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
