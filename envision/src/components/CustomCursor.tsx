import { useEffect, useState } from 'react';
import { motion } from 'motion/react';

export default function CustomCursor() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);
  const [isHidden, setIsHidden] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    if (imageFailed) {
      document.body.classList.add('fallback-cursor');
    } else {
      document.body.classList.remove('fallback-cursor');
    }
    return () => document.body.classList.remove('fallback-cursor');
  }, [imageFailed]);

  useEffect(() => {
    const updateMousePosition = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName.toLowerCase() === 'button' ||
        target.tagName.toLowerCase() === 'a' ||
        target.closest('button') ||
        target.closest('a') ||
        target.classList.contains('cursor-pointer')
      ) {
        setIsHovering(true);
      } else {
        setIsHovering(false);
      }

      if (target.tagName.toLowerCase() === 'input' || target.tagName.toLowerCase() === 'textarea') {
        setIsHidden(true);
      } else {
        setIsHidden(false);
      }
    };

    const handleMouseLeave = () => setIsHidden(true);
    const handleMouseEnter = () => setIsHidden(false);

    window.addEventListener('mousemove', updateMousePosition);
    window.addEventListener('mouseover', handleMouseOver);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);

    return () => {
      window.removeEventListener('mousemove', updateMousePosition);
      window.removeEventListener('mouseover', handleMouseOver);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
    };
  }, []);

  if (imageFailed) return null;

  return (
    <motion.div
      className="fixed top-0 left-0 pointer-events-none z-[9999]"
      style={{
        x: mousePosition.x,
        y: mousePosition.y,
        opacity: isHidden ? 0 : 1,
      }}
      animate={{
        scale: isHovering ? 1.1 : 1,
        rotate: isHovering ? -10 : 0,
      }}
      transition={{
        type: 'spring',
        stiffness: 500,
        damping: 28,
        mass: 0.5,
      }}
    >
      <img
        src="/pencilcurs.png"
        alt="cursor"
        className="w-8 h-8 object-contain origin-top-left"
        style={{ transform: 'translate(-10%, -10%)' }}
        onError={() => setImageFailed(true)}
      />
    </motion.div>
  );
}
