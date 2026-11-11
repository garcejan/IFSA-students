"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform, type MotionValue } from "motion/react";

const TEXT =
  "Seven agents run a diagnostics department. A seizure goes to Foreman. A lawsuit goes to Cuddy. The medicine is synthetic and deliberately shallow. The orchestration is real.";

function Word({ word, range, progress }: { word: string; range: [number, number]; progress: MotionValue<number> }) {
  const opacity = useTransform(progress, range, [0.12, 1]);
  const y = useTransform(progress, range, [10, 0]);
  return (
    <motion.span style={{ opacity, y }} className="mr-[0.28em] inline-block">
      {word}
    </motion.span>
  );
}

export default function Manifesto() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 0.8", "end 0.55"] });
  const words = TEXT.split(" ");

  return (
    <section id="manifesto" ref={ref} className="relative bg-paper px-6 py-40">
      <div className="mx-auto max-w-5xl">
        <p className="mb-8 font-mono text-xs uppercase tracking-[0.25em] text-teal">Our mission</p>
        <p className="text-[clamp(1.9rem,4.6vw,3.9rem)] font-medium leading-[1.12] tracking-[-0.025em]">
          {words.map((w, i) => {
            const start = (i / words.length) * 0.9;
            return <Word key={i} word={w} range={[start, start + 0.12]} progress={scrollYProgress} />;
          })}
        </p>
      </div>
    </section>
  );
}
