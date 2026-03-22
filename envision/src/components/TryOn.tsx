import { useState, useEffect, useRef, type ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Shirt, Loader2, AlertCircle, Upload, ChevronLeft, ChevronRight } from 'lucide-react';
import { getCartItems, removeCartItem, tryOnGarment, cachePerson, categoryToTab, tabToCategory, type CartItem } from '../utils/api';

interface FitVariant { label: string; img: string; }

interface ClosetItem {
  id: number;
  name: string;
  img: string;
  garment_url: string;
  category: 'tops' | 'pants';
  hero?: boolean;
  beforeImg?: string;
  afterImg?: string;
  fitVariants?: FitVariant[];  // T/F/B switcher (pants)
}

interface TryOnProps {
  userData: any;
  sessionId: string;
}

// ── Demo shopping item (hardcoded, appears first in Shopping > Tops) ─────────
const DEMO_SHOPPING_ITEMS: ClosetItem[] = [
  { id: 20, name: 'Element Shirt', img: '/element_sticker.png', garment_url: 'http://localhost:8000/garments/elementshirt.png', category: 'tops', hero: true, beforeImg: '/nikeshirtwhitebackground.png', afterImg: '/perfectelement.jpg' },
];

// ── Hardcoded closet items ────────────────────────────────────────────────────
const CLOSET_ITEMS: ClosetItem[] = [
  { id: 10, name: 'Diary Of Wimpy Kid', img: '/garment_sticker.png',        garment_url: 'http://localhost:8000/garments/garment_clean.jpg',  category: 'tops',  hero: true, beforeImg: '/mexico.jpg', afterImg: '/dowk_helena.jpg' },
  { id: 11, name: 'Late Registration',  img: '/latereg_tshirt_sticker.png', garment_url: 'http://localhost:8000/garments/latereg_tshirt.jpg', category: 'tops',  hero: true, beforeImg: '/dcsiz.jpg',  afterImg: '/latereg_good.jpg' },
  { id: 12, name: 'Black Double Knees', img: '/blackdoubleknee_sticker.png', garment_url: 'http://localhost:8000/garments/blackdoubleknee.png', category: 'pants', hero: true, beforeImg: '/jeanchud.png',
    fitVariants: [
      { label: 'T', img: '/result_tight.jpg' },
      { label: 'F', img: '/result_fitted.jpg' },
      { label: 'B', img: '/result_loose.jpg' },
    ],
  },
];

// ── Before/After Slider ───────────────────────────────────────────────────────
function BeforeAfterSlider({ beforeImg, afterImg }: { beforeImg: string; afterImg: string }) {
  const [pos, setPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const move = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setPos(Math.max(2, Math.min(98, ((clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none cursor-col-resize"
      onMouseMove={e => dragging && move(e.clientX)}
      onMouseDown={e => { setDragging(true); move(e.clientX); }}
      onMouseUp={() => setDragging(false)}
      onMouseLeave={() => setDragging(false)}
      onTouchMove={e => move(e.touches[0].clientX)}
      onTouchStart={e => move(e.touches[0].clientX)}
    >
      <img src={afterImg} alt="AI Result" className="absolute inset-0 w-full h-full object-cover" draggable={false} />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={beforeImg} alt="Original" className="w-full h-full object-cover" draggable={false} />
      </div>
      <div className="absolute top-0 bottom-0 z-20 pointer-events-none" style={{ left: `${pos}%`, transform: 'translateX(-50%)' }}>
        <div className="w-0.5 h-full bg-white shadow-[0_0_6px_rgba(0,0,0,0.6)]" />
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-[14px] w-8 h-8 bg-white rounded-full shadow-[0_2px_12px_rgba(0,0,0,0.4)] flex items-center justify-center">
          <ChevronLeft size={12} className="text-royal-blue/60" />
          <ChevronRight size={12} className="text-royal-blue/60" />
        </div>
      </div>
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 flex gap-2 pointer-events-none">
        <span className="bg-black/50 text-white font-pixel text-[8px] px-2 py-0.5 rounded">ORIGINAL</span>
        <span className="bg-envision-pink/80 text-white font-pixel text-[8px] px-2 py-0.5 rounded">AI TRY-ON</span>
      </div>
    </div>
  );
}

// ── Fit Switcher (T / F / B) ──────────────────────────────────────────────────
function FitSwitcherSlider({ beforeImg, fitVariants }: { beforeImg: string; fitVariants: FitVariant[] }) {
  const [activeFit, setActiveFit] = useState(0);
  const [pos, setPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const move = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setPos(Math.max(2, Math.min(98, ((clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none cursor-col-resize"
      onMouseMove={e => dragging && move(e.clientX)}
      onMouseDown={e => { setDragging(true); move(e.clientX); }}
      onMouseUp={() => setDragging(false)}
      onMouseLeave={() => setDragging(false)}
      onTouchMove={e => move(e.touches[0].clientX)}
      onTouchStart={e => move(e.touches[0].clientX)}
    >
      <img src={fitVariants[activeFit].img} alt="AI Result" className="absolute inset-0 w-full h-full object-cover" draggable={false} />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={beforeImg} alt="Original" className="w-full h-full object-cover" draggable={false} />
      </div>
      {/* Draggable divider */}
      <div className="absolute top-0 bottom-0 z-20 pointer-events-none" style={{ left: `${pos}%`, transform: 'translateX(-50%)' }}>
        <div className="w-0.5 h-full bg-white shadow-[0_0_6px_rgba(0,0,0,0.6)]" />
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-[14px] w-8 h-8 bg-white rounded-full shadow-[0_2px_12px_rgba(0,0,0,0.4)] flex items-center justify-center">
          <ChevronLeft size={12} className="text-royal-blue/60" />
          <ChevronRight size={12} className="text-royal-blue/60" />
        </div>
      </div>
      <div className="absolute bottom-12 left-1/2 -translate-x-1/2 z-10 flex gap-2 pointer-events-none">
        <span className="bg-black/50 text-white font-pixel text-[8px] px-2 py-0.5 rounded">ORIGINAL</span>
        <span className="bg-envision-pink/80 text-white font-pixel text-[8px] px-2 py-0.5 rounded">AI TRY-ON</span>
      </div>
      {/* T / F / B buttons */}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-30 flex gap-2 pointer-events-auto">
        {fitVariants.map((v, i) => (
          <button
            key={v.label}
            onMouseDown={e => e.stopPropagation()}
            onClick={e => { e.stopPropagation(); setActiveFit(i); }}
            className={`w-8 h-8 rounded-full font-space-mono text-xs font-bold transition-all duration-200 ${
              activeFit === i
                ? 'bg-envision-pink text-white shadow-[0_2px_8px_rgba(0,0,0,0.4)] scale-110'
                : 'bg-black/40 text-white/70 hover:bg-black/60'
            }`}
          >{v.label}</button>
        ))}
      </div>
    </div>
  );
}

export default function TryOn({ userData, sessionId }: TryOnProps) {
  const [activeMainTab, setActiveMainTab] = useState<'closet' | 'shopping'>('closet');
  const [activeCategory, setActiveCategory] = useState<'tops' | 'pants'>('tops');

  const [shoppingItems, setShoppingItems] = useState<CartItem[]>([]);
  const [loadingShop, setLoadingShop] = useState(false);

  const photoInputRef = useRef<HTMLInputElement>(null);
  const [personImage, setPersonImage] = useState<string | null>(userData.modelImage ?? null);
  const [uploadingPhoto, setUploadingPhoto] = useState(false);

  const handlePhotoSelect = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPersonImage(URL.createObjectURL(file));
    setResultImage(null);
    setUploadingPhoto(true);
    try { await cachePerson(sessionId, file); } finally { setUploadingPhoto(false); }
  };

  const [tryingOn, setTryingOn] = useState(false);
  const [tryOnError, setTryOnError] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [activeItemId, setActiveItemId] = useState<string | number | null>(null);

  const heroItems = [...CLOSET_ITEMS, ...DEMO_SHOPPING_ITEMS].filter(i => i.hero && i.beforeImg && (i.afterImg || i.fitVariants));
  const [activeHeroId, setActiveHeroId] = useState<number>(heroItems[0]?.id ?? 10);
  const activeHero = heroItems.find(i => i.id === activeHeroId) ?? heroItems[0];

  useEffect(() => {
    if (activeMainTab !== 'shopping') return;
    setLoadingShop(true);
    getCartItems(sessionId).then(items => { setShoppingItems(items); setLoadingShop(false); });
  }, [activeMainTab, sessionId]);

  const handleRemove = async (itemId: string) => {
    await removeCartItem(sessionId, itemId);
    setShoppingItems(prev => prev.filter(i => i.id !== itemId));
  };

  const handleTryOn = async (garmentUrl: string, category: 'tops' | 'pants', itemId: string | number) => {
    if (tryingOn) return;
    setTryingOn(true); setTryOnError(null); setActiveItemId(itemId);
    try {
      const url = await tryOnGarment(sessionId, garmentUrl, tabToCategory(category));
      setResultImage(url);
    } catch (err: any) {
      setTryOnError(err.message ?? 'Try-on failed');
    } finally {
      setTryingOn(false);
    }
  };

  type DisplayItem = { id: number | string; name: string; img: string; garment_url: string; category: 'tops' | 'pants'; hero?: boolean };
  const closetDisplay = CLOSET_ITEMS.filter(i => i.category === activeCategory);
  const shoppingDisplay = [
    ...DEMO_SHOPPING_ITEMS.filter(i => i.category === activeCategory),
    ...shoppingItems.filter(i => categoryToTab(i.category) === activeCategory),
  ];
  const displayItems: DisplayItem[] = activeMainTab === 'closet'
    ? closetDisplay.map(i => ({ id: i.id, name: i.name, img: i.img, garment_url: i.garment_url, category: i.category, hero: i.hero }))
    : shoppingDisplay.map(i => {
        const img = 'img' in i ? i.img : (i as CartItem).image_url;
        const garment = 'garment_url' in i ? i.garment_url : (i as CartItem).image_url;
        const isCloset = 'img' in i;
        return { id: i.id, name: i.name, img, garment_url: garment, category: categoryToTab(i.category) as 'tops' | 'pants', hero: isCloset ? (i as ClosetItem).hero : false };
      });

  // Determine what to show in the left panel
  const showResult = !tryingOn && !tryOnError && resultImage;
  const showPerson = !tryingOn && !tryOnError && !resultImage && personImage;


  return (
    <div className="h-screen w-full flex pt-16 relative overflow-hidden">
      <input ref={photoInputRef} type="file" accept="image/*" className="hidden" onChange={handlePhotoSelect} />

      {/* Left Side */}
      <div className="w-1/2 h-full flex items-center justify-center p-12">
        <motion.div
          initial={{ opacity: 0, rotate: -2, scale: 0.9 }}
          animate={{ opacity: 1, rotate: 0, scale: 1 }}
          className="relative w-full max-w-lg aspect-[3/4] paper-texture p-8 shadow-[0_20px_40px_rgba(0,0,0,0.15)]"
        >
          <img src="/pinktape.png" alt="" className="absolute w-32 -top-6 -left-8 -rotate-12 z-20 drop-shadow-md pointer-events-none object-contain" />
          <img src="/pinktape.png" alt="" className="absolute w-32 -top-6 -right-8 rotate-12 z-20 drop-shadow-md pointer-events-none object-contain" />
          <img src="/pinktape.png" alt="" className="absolute w-32 -bottom-6 -left-8 rotate-45 z-20 drop-shadow-md pointer-events-none object-contain" />
          <img src="/pinktape.png" alt="" className="absolute w-32 -bottom-6 -right-8 -rotate-45 z-20 drop-shadow-md pointer-events-none object-contain" />

          <div className="w-full h-full bg-white/60 border border-black/5 flex items-center justify-center relative overflow-hidden shadow-inner">
            {tryingOn ? (
              <div className="flex flex-col items-center gap-4 text-royal-blue/50">
                <Loader2 size={48} className="animate-spin text-envision-pink" />
                <p className="font-ubuntu font-bold tracking-widest text-xs">GENERATING TRY-ON...</p>
                <p className="font-mono text-[10px] text-royal-blue/30">this takes ~30 seconds</p>
              </div>
            ) : tryOnError ? (
              <div className="flex flex-col items-center gap-4 text-red-400 px-4 text-center">
                <AlertCircle size={40} />
                <p className="font-mono text-xs">{tryOnError}</p>
                <button onClick={() => setTryOnError(null)} className="text-[10px] font-pixel text-envision-pink hover:underline">DISMISS</button>
              </div>
            ) : showResult ? (
              <div className="relative w-full h-full">
                <img src={resultImage!} alt="Result" className="w-full h-full object-cover" />
                <button
                  onClick={() => setResultImage(null)}
                  className="absolute top-2 right-2 bg-black/50 text-white text-[10px] font-pixel px-2 py-1 rounded hover:bg-black/70 z-10"
                >RESET</button>
              </div>
            ) : showPerson ? (
              <div className="relative w-full h-full group">
                <img src={personImage!} alt="Model" className="w-full h-full object-cover" />
                <div
                  onClick={() => photoInputRef.current?.click()}
                  className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-300 flex items-center justify-center cursor-pointer"
                >
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center gap-2">
                    <Upload size={28} className="text-white" />
                    <span className="font-pixel text-[9px] text-white tracking-widest">CHANGE PHOTO</span>
                  </div>
                </div>
                {uploadingPhoto && (
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                    <Loader2 size={32} className="animate-spin text-white" />
                  </div>
                )}
              </div>
            ) : (
              /* Before/After or Fit Switcher — default state */
              activeHero?.fitVariants
                ? <FitSwitcherSlider beforeImg={activeHero.beforeImg!} fitVariants={activeHero.fitVariants} />
                : <BeforeAfterSlider beforeImg={activeHero.beforeImg!} afterImg={activeHero.afterImg!} />
            )}

            {/* Upload button always accessible when not showing result */}
            {!resultImage && !tryingOn && !tryOnError && (
              <button
                onClick={() => photoInputRef.current?.click()}
                className="absolute top-2 right-2 z-30 bg-black/40 hover:bg-envision-pink/90 text-white font-pixel text-[8px] px-2 py-1 rounded transition-all flex items-center gap-1"
              >
                <Upload size={10} />
                UPLOAD
              </button>
            )}
          </div>
        </motion.div>
      </div>

      {/* Shared SVG filter for sticker border — defined once */}
      <svg width="0" height="0" className="absolute pointer-events-none">
        <defs>
          <filter id="sticker-border" x="-10%" y="-10%" width="120%" height="120%">
            <feMorphology operator="dilate" radius="6" in="SourceAlpha" result="dilated" />
            <feFlood floodColor="white" result="white" />
            <feComposite in="white" in2="dilated" operator="in" result="outline" />
            <feDropShadow dx="0" dy="6" stdDeviation="8" floodColor="rgba(0,0,0,0.45)" />
            <feMerge>
              <feMergeNode in="outline" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>

      {/* Right Side: Closet / Shopping */}
      <div className="w-1/2 h-full flex flex-col pt-0 bg-white/40 backdrop-blur-md border-l border-royal-blue/10 shadow-2xl">
        {/* Main Tabs */}
        <div className="w-full mb-0 overflow-hidden py-4 -mt-6 -mb-4">
          <div className="flex justify-between w-full relative z-50 pointer-events-none">
            <button
              onClick={() => setActiveMainTab('closet')}
              className={`pointer-events-auto relative transition-all duration-300 ease-out flex items-center justify-end pr-6 w-72 h-24 ${activeMainTab === 'closet' ? '-translate-x-4' : '-translate-x-12 hover:-translate-x-8'}`}
            >
              <img src="/leftlabel.png" alt="Closet" className="absolute inset-0 w-[115%] max-w-none h-full object-fill drop-shadow-md -left-[15%]" />
              <span className={`relative z-10 font-gloria font-bold tracking-tighter text-2xl ${activeMainTab === 'closet' ? 'text-envision-pink' : 'text-black/50'}`}>CLOSET</span>
            </button>
            <button
              onClick={() => setActiveMainTab('shopping')}
              className={`pointer-events-auto relative transition-all duration-300 ease-out flex items-center justify-start pl-6 w-72 h-24 ${activeMainTab === 'shopping' ? 'translate-x-4' : 'translate-x-12 hover:translate-x-8'}`}
            >
              <img src="/rightlabel.png" alt="Shopping" className="absolute inset-0 w-full h-full object-fill drop-shadow-md" />
              <span className={`relative z-10 font-gloria font-bold tracking-tighter text-2xl ${activeMainTab === 'shopping' ? 'text-envision-pink' : 'text-black/50'}`}>SHOPPING</span>
            </button>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex gap-16 mb-8 -mt-8 items-center justify-center">
          {(['tops', 'pants'] as const).map(cat => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`relative group transition-all duration-300 ease-out flex flex-col items-center ${activeCategory === cat ? 'opacity-100 scale-105' : 'opacity-60 hover:opacity-100 hover:scale-105'}`}
            >
              <img src={cat === 'tops' ? '/shirtbutt.png' : '/pantsbutt.png'} alt={cat} className="w-20 h-20 object-contain drop-shadow-md" />
              <span className={`absolute -bottom-5 font-gloria text-lg font-bold text-[rgb(152,20,60)] transition-opacity duration-300 pointer-events-none whitespace-nowrap ${activeCategory === cat ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
                {cat === 'tops' ? 'Tops' : 'Bottoms'}
              </span>
              {activeCategory === cat && (
                <motion.img layoutId="categoryIndicator" src="/redline.png" alt="" className="absolute -bottom-9 w-24 h-3 object-contain" />
              )}
            </button>
          ))}
        </div>

        {/* Items Grid */}
        <div
          className="flex-1 overflow-y-auto mt-8 custom-scrollbar relative"
          style={{
            backgroundImage: activeMainTab === 'closet' ? 'url(/twoshelf.jpg)' : 'url(/horbasket.png)',
            backgroundSize: '100% 100%',
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'center',
          }}
        >
          {loadingShop ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 size={32} className="animate-spin text-envision-pink" />
            </div>
          ) : displayItems.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 opacity-40">
              <Shirt size={32} className="text-royal-blue" />
              <p className="font-mono text-xs text-royal-blue tracking-widest">
                {activeMainTab === 'shopping' ? 'ADD ITEMS VIA EXTENSION' : 'NO ITEMS'}
              </p>
            </div>
          ) : (
            <div className={`min-h-full px-8 pb-8 pt-16 grid grid-cols-2 ${activeMainTab === 'closet' ? 'gap-x-8 gap-y-16 content-start' : 'gap-x-8 gap-y-8 content-center'}`}>
              <AnimatePresence mode="popLayout">
                {displayItems.map((item) => (
                  item.hero ? (
                    /* ── Hero / Sticker item ── */
                    <motion.div
                      key={item.id}
                      layout
                      initial={{ opacity: 0, scale: 0.8, rotate: -8 }}
                      animate={{ opacity: 1, scale: 1, rotate: -3 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      whileHover={{ rotate: [-3, -6, 1, -6, 1, -3], transition: { duration: 0.5, ease: 'easeInOut' } }}
                      onClick={() => setActiveHeroId(item.id as number)}
                      className="col-span-2 flex flex-col items-center gap-3 cursor-pointer"
                    >
                      <img
                        src={item.img}
                        alt={item.name}
                        className="w-40 aspect-square object-contain"
                        style={{ filter: 'url(#sticker-border)' }}
                      />
                      <p className="font-space-mono text-[10px] text-black tracking-wider text-center -mt-2">{item.name}</p>
                    </motion.div>
                  ) : (
                    /* ── Regular / Shopping item — sticker style ── */
                    <motion.div
                      key={item.id}
                      layout
                      initial={{ opacity: 0, scale: 0.8, rotate: -8 }}
                      animate={{ opacity: 1, scale: 1, rotate: activeItemId === item.id && tryingOn ? 0 : -2 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      whileHover={activeItemId === item.id && tryingOn ? {} : { rotate: [-2, -5, 1, -5, 1, -2], transition: { duration: 0.5, ease: 'easeInOut' } }}
                      className={`col-span-1 flex flex-col items-center gap-3 cursor-pointer relative ${activeItemId === item.id && tryingOn ? 'opacity-50 pointer-events-none' : ''}`}
                      onClick={() => handleTryOn(item.garment_url, item.category, item.id)}
                    >
                      <div className="relative">
                        <img
                          src={item.img}
                          alt={item.name}
                          className="w-28 aspect-square object-contain"
                          style={{ filter: 'url(#sticker-border)' }}
                          referrerPolicy="no-referrer"
                        />
                        <button
                          onMouseDown={e => e.stopPropagation()}
                          onClick={e => { e.stopPropagation(); handleRemove(item.id as string); }}
                          className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-red-500 text-white text-xs font-bold flex items-center justify-center shadow-md hover:bg-red-600 transition-colors z-10"
                        >−</button>
                      </div>
                      <p className="font-space-mono text-[9px] text-black tracking-wider text-center -mt-2 leading-tight">{item.name}</p>
                    </motion.div>
                  )
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
