import { useState, useRef, type ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronRight, ChevronLeft, Upload, Settings } from 'lucide-react';
import { cachePerson } from '../utils/api';

interface PreferencesProps {
  sessionId: string;
  onComplete: (data: any) => void;
}

export default function Preferences({ sessionId, onComplete }: PreferencesProps) {
  const [step, setStep] = useState(1);
  const [gender, setGender] = useState<'male' | 'female' | 'unisex' | ''>('');
  const [topSize, setTopSize] = useState('');
  const [bottomSize, setBottomSize] = useState('');
  const [bottomLength, setBottomLength] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [measurements, setMeasurements] = useState({
    chest: '',
    armLength: '',
    waist: '',
    inseam: '',
  });
  const [username, setUsername] = useState('');

  // Step 3: photo upload
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const topsSizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL'];
  const femalePants = ['00', '0', '2', '4', '6', '8', '10', '12', '14', '16'];
  const maleWaist = ['28', '30', '32', '34', '36', '38', '40'];

  const handleGenderSelect = (g: 'male' | 'female' | 'unisex') => {
    setGender(g);
    if (g === 'unisex') {
      setShowAdvanced(true);
    }
    setTimeout(() => setStep(2), 400);
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setModelFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setUploadError(null);
  };

  const handleNext = async () => {
    if (step < 3) { setStep(step + 1); return; }

    // Step 3 — upload photo then complete
    setUploading(true);
    setUploadError(null);
    try {
      if (modelFile) {
        await cachePerson(sessionId, modelFile);
      }
      onComplete({ gender, topSize, bottomSize, bottomLength, measurements, username, modelImage: previewUrl });
    } catch (err: any) {
      setUploadError(err.message ?? 'Upload failed, please try again.');
      setUploading(false);
    }
  };

  const handleBack = () => {
    if (step > 1) setStep(step - 1);
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center py-16 relative">
      {/* Grid Lines are global */}

      <div className="max-w-4xl w-full px-8 space-y-12 z-10 flex flex-col items-center relative">
        {step > 1 && (
          <button
            onClick={handleBack}
            className="absolute top-2 left-4 md:top-4 md:left-0 w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-[inset_0_-3px_6px_rgba(0,0,0,0.2),0_3px_6px_rgba(0,0,0,0.3)] hover:scale-105 active:scale-95 transition-all text-royal-blue z-50"
          >
            <ChevronLeft size={24} className="-ml-1" />
          </button>
        )}

        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="ripped-paper mb-8"
        >
          <h2 className="text-5xl font-bold font-ubuntu tracking-tighter text-center text-envision-pink">
            preferences
          </h2>
        </motion.div>

        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-12 w-full"
            >
              <div className="flex justify-center gap-8">
                {[
                   { id: 'female', label: 'FEMALE' },
                   { id: 'male', label: 'MALE' },
                   { id: 'unisex', label: 'UNISEX' }
                ].map((g) => (
                  <button
                    key={g.id}
                    onClick={() => handleGenderSelect(g.id as any)}
                    className={`washi-tape-button min-w-[160px] !text-lg ${
                      gender === g.id
                        ? ''
                        : 'opacity-60 grayscale'
                    }`}
                  >
                    {g.label}
                  </button>
                ))}
              </div>

              {/* Sizing options moved to step 2 */}
            </motion.div>
          )}

          {step === 2 && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-8 w-full max-h-[60vh] overflow-y-auto subtle-scrollbar pr-4 pb-4"
            >
              {gender && gender !== 'unisex' && (
                <div className="space-y-8 mb-12">
                  <div className="space-y-4">
                    <p className="text-xs font-mono tracking-widest uppercase text-royal-blue/50 text-center">TOPS SIZING</p>
                    <div className="flex flex-wrap justify-center gap-4">
                      {topsSizes.map(size => (
                        <button
                          key={size}
                          onClick={() => setTopSize(size)}
                          className={`relative w-24 h-24 flex items-center justify-center transition-all duration-300 hover:scale-105 active:scale-95 ${
                            topSize === size ? 'scale-110 drop-shadow-[0_0_15px_rgba(219,85,139,0.6)]' : 'opacity-80 hover:opacity-100'
                          }`}
                        >
                          <img
                            src="/whbutton.png"
                            alt="button border"
                            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                          />
                          <div className="relative z-10 w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-[inset_0_-3px_6px_rgba(0,0,0,0.2),0_3px_6px_rgba(0,0,0,0.3)]">
                            <span className="font-ubuntu font-bold text-royal-blue text-sm">{size}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-4">
                    {gender === 'female' ? (
                      <>
                        <p className="text-xs font-mono tracking-widest uppercase text-royal-blue/50 text-center">PANTS SIZING</p>
                        <div className="flex flex-wrap justify-center gap-4">
                          {femalePants.map(size => (
                            <button
                              key={size}
                              onClick={() => setBottomSize(size)}
                              className={`relative w-24 h-24 flex items-center justify-center transition-all duration-300 hover:scale-105 active:scale-95 ${
                                bottomSize === size ? 'scale-110 drop-shadow-[0_0_15px_rgba(219,85,139,0.6)]' : 'opacity-80 hover:opacity-100'
                              }`}
                            >
                              <img
                                src="/pinbutton.png"
                                alt="button border"
                                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                              />
                              <div className="relative z-10 w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-[inset_0_-3px_6px_rgba(0,0,0,0.2),0_3px_6px_rgba(0,0,0,0.3)]">
                                <span className="font-ubuntu font-bold text-royal-blue text-sm">{size}</span>
                              </div>
                            </button>
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="space-y-8">
                        <div className="space-y-4">
                          <p className="text-xs font-mono tracking-widest uppercase text-royal-blue/50 text-center">WAIST SIZING</p>
                          <div className="flex flex-wrap justify-center gap-4">
                            {maleWaist.map(size => (
                              <button
                                key={`waist-${size}`}
                                onClick={() => setBottomSize(size)}
                                className={`relative w-24 h-24 flex items-center justify-center transition-all duration-300 hover:scale-105 active:scale-95 ${
                                  bottomSize === size ? 'scale-110 drop-shadow-[0_0_15px_rgba(219,85,139,0.6)]' : 'opacity-80 hover:opacity-100'
                                }`}
                              >
                                <img
                                  src="/pinbutton.png"
                                  alt="button border"
                                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                                />
                                <div className="relative z-10 w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-[inset_0_-3px_6px_rgba(0,0,0,0.2),0_3px_6px_rgba(0,0,0,0.3)]">
                                  <span className="font-ubuntu font-bold text-royal-blue text-sm">{size}</span>
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="space-y-4">
                          <p className="text-xs font-mono tracking-widest uppercase text-royal-blue/50 text-center">LENGTH SIZING</p>
                          <div className="flex flex-wrap justify-center gap-4">
                            {maleWaist.map(size => (
                              <button
                                key={`length-${size}`}
                                onClick={() => setBottomLength(size)}
                                className={`relative w-24 h-24 flex items-center justify-center transition-all duration-300 hover:scale-105 active:scale-95 ${
                                  bottomLength === size ? 'scale-110 drop-shadow-[0_0_15px_rgba(219,85,139,0.6)]' : 'opacity-80 hover:opacity-100'
                                }`}
                              >
                                <img
                                  src="/bluebutton.png"
                                  alt="button border"
                                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                                />
                                <div className="relative z-10 w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-[inset_0_-3px_6px_rgba(0,0,0,0.2),0_3px_6px_rgba(0,0,0,0.3)]">
                                  <span className="font-ubuntu font-bold text-royal-blue text-sm">{size}</span>
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between">
                <p className="text-xl font-ubuntu font-bold tracking-widest text-royal-blue">ALTERATIONS / SPECIFICATIONS</p>
                {gender !== 'unisex' && (
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-envision-pink hover:text-royal-blue transition-all font-pixel text-[10px]"
                  >
                    <Settings size={20} />
                    {showAdvanced ? 'HIDE' : 'EDIT'}
                  </button>
                )}
              </div>

              {(showAdvanced || gender === 'unisex') && (
                <div className="grid grid-cols-2 gap-8 bg-white/40 backdrop-blur-sm border border-royal-blue/10 p-8 rounded-2xl shadow-xl">
                  <div className="space-y-2">
                    <label className="text-xs font-mono uppercase text-royal-blue/50">CHEST (IN/CM)</label>
                    <input
                      type="text"
                      className="w-full bg-white/50 border border-royal-blue/10 rounded-xl px-4 py-3 focus:border-envision-pink outline-none font-mono text-royal-blue"
                      placeholder="e.g. 38"
                      value={measurements.chest}
                      onChange={e => setMeasurements({...measurements, chest: e.target.value})}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-mono uppercase text-royal-blue/50">ARM LENGTH</label>
                    <input
                      type="text"
                      className="w-full bg-white/50 border border-royal-blue/10 rounded-xl px-4 py-3 focus:border-envision-pink outline-none font-mono text-royal-blue"
                      placeholder="e.g. 24"
                      value={measurements.armLength}
                      onChange={e => setMeasurements({...measurements, armLength: e.target.value})}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-mono uppercase text-royal-blue/50">WAIST</label>
                    <input
                      type="text"
                      className="w-full bg-white/50 border border-royal-blue/10 rounded-xl px-4 py-3 focus:border-envision-pink outline-none font-mono text-royal-blue"
                      placeholder="e.g. 32"
                      value={measurements.waist}
                      onChange={e => setMeasurements({...measurements, waist: e.target.value})}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-mono uppercase text-royal-blue/50">INSEAM</label>
                    <input
                      type="text"
                      className="w-full bg-white/50 border border-royal-blue/10 rounded-xl px-4 py-3 focus:border-envision-pink outline-none font-mono text-royal-blue"
                      placeholder="e.g. 30"
                      value={measurements.inseam}
                      onChange={e => setMeasurements({...measurements, inseam: e.target.value})}
                    />
                  </div>
                </div>
              )}

              <div className="space-y-4">
                <p className="text-sm font-mono tracking-widest uppercase text-royal-blue/50">SAVE FOR NEXT TIME (OPTIONAL)</p>
                <input
                  type="text"
                  className="w-full bg-white/50 border border-royal-blue/10 rounded-2xl px-8 py-6 focus:border-envision-pink outline-none text-xl font-ubuntu font-bold text-royal-blue"
                  placeholder="USERNAME"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                />
              </div>

              <div className="flex justify-center pt-8">
                <button
                  onClick={() => setStep(3)}
                  className="washi-tape-button !text-lg !px-8 !py-3"
                >
                  CONTINUE <ChevronRight size={20} />
                </button>
              </div>
            </motion.div>
          )}

          {step === 3 && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-12 flex flex-col items-center w-full max-h-[60vh] overflow-y-auto subtle-scrollbar pr-4 pb-4"
            >
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />

              <div
                onClick={() => fileInputRef.current?.click()}
                className="w-full max-w-md aspect-[3/4] bg-white/40 backdrop-blur-sm border-dashed border-2 border-royal-blue/20 rounded-3xl flex flex-col items-center justify-center gap-6 hover:border-envision-pink transition-all cursor-pointer group shadow-xl overflow-hidden"
              >
                {previewUrl ? (
                  <img src={previewUrl} alt="Your photo" className="w-full h-full object-cover" />
                ) : (
                  <>
                    <div className="w-24 h-24 rounded-full bg-royal-blue/5 flex items-center justify-center group-hover:bg-envision-pink/20 transition-all">
                      <Upload size={48} className="text-royal-blue/30 group-hover:text-envision-pink" />
                    </div>
                    <div className="text-center">
                      <p className="text-xl font-ubuntu font-bold tracking-widest text-royal-blue">UPLOAD MODEL IMAGE</p>
                      <p className="text-xs font-mono text-royal-blue/40 mt-2 uppercase">FULL BODY, NEUTRAL BACKGROUND</p>
                    </div>
                  </>
                )}
              </div>

              {previewUrl && (
                <p className="text-xs font-mono text-envision-pink tracking-widest">PHOTO READY — CLICK TO CHANGE</p>
              )}
              {uploadError && (
                <p className="text-xs font-mono text-red-500 tracking-widest">{uploadError}</p>
              )}

              <div className="flex justify-center pt-4">
                <button
                  onClick={handleNext}
                  disabled={uploading}
                  className="washi-tape-button !text-lg !px-8 !py-3"
                >
                  {uploading ? 'UPLOADING...' : 'GO TO CLOSET'} <ChevronRight size={20} />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
}
