"use client";

import { useRef } from "react";
import {
  animate,
  motion,
  useInView,
  useMotionValue,
  useScroll,
  useTransform,
} from "motion/react";
import { useEffect } from "react";

const ECG =
  "M0 150 H180 L205 150 L222 110 L240 150 H300 L318 165 L345 20 L378 250 L402 150 H470 L492 128 L520 150 H700 L725 150 L742 112 L760 150 H820 L838 165 L865 30 L898 245 L922 150 H990 L1012 130 L1040 150 H1200";

function Counter({ to, label }: { to: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true });
  const v = useMotionValue(0);
  const rounded = useTransform(v, (n) => Math.round(n));

  useEffect(() => {
    if (!inView) return;
    const c = animate(v, to, { duration: 1.6, ease: "easeOut", delay: 1.2 });
    return () => c.stop();
  }, [inView, to, v]);

  return (
    <div ref={ref} className="text-left">
      <motion.div className="font-mono text-4xl font-medium tracking-tight sm:text-5xl">{rounded}</motion.div>
      <div className="mt-1 text-xs uppercase tracking-[0.2em] text-muted">{label}</div>
    </div>
  );
}

const WORDS = ["Everybody", "lies."];
const WORDS2 = ["The", "chart", "doesn’t."];

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const titleY = useTransform(scrollYProgress, [0, 1], [0, 160]);
  const ecgY = useTransform(scrollYProgress, [0, 1], [0, 80]);
  const fade = useTransform(scrollYProgress, [0, 0.7], [1, 0]);
  const orbY = useTransform(scrollYProgress, [0, 1], [0, -120]);

  return (
    <section id="top" ref={ref} className="grid-paper relative flex min-h-[110vh] items-start overflow-hidden pt-36">
      {/* aurora */}
      <motion.div
        style={{ y: orbY }}
        className="pointer-events-none absolute -right-40 top-10 h-[640px] w-[640px] rounded-full bg-[radial-gradient(circle,rgba(25,195,230,0.38),rgba(0,167,157,0.12)_55%,transparent_70%)] blur-2xl"
      />
      <div className="pointer-events-none absolute -left-52 top-1/2 h-[520px] w-[520px] rounded-full bg-[radial-gradient(circle,rgba(42,109,244,0.18),transparent_65%)] blur-2xl" />

      <motion.div style={{ y: titleY, opacity: fade }} className="relative mx-auto w-full max-w-6xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8 inline-flex items-center gap-2.5 rounded-full border border-line bg-white/70 px-4 py-1.5 font-mono text-[11px] uppercase tracking-[0.2em] text-ink-2"
        >
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-teal opacity-70" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-teal" />
          </span>
          Department of Diagnostic Medicine · live
        </motion.div>

        <h1 className="text-[clamp(3rem,10vw,8.5rem)] font-semibold leading-[0.92] tracking-[-0.04em]">
          <span className="block overflow-hidden pb-2">
            {WORDS.map((w, i) => (
              <motion.span
                key={w}
                initial={{ y: "110%" }}
                animate={{ y: 0 }}
                transition={{ duration: 0.9, delay: 0.35 + i * 0.1, ease: [0.22, 1, 0.36, 1] }}
                className="mr-[0.25em] inline-block"
              >
                {w}
              </motion.span>
            ))}
          </span>
          <span className="block overflow-hidden pb-3">
            {WORDS2.map((w, i) => (
              <motion.span
                key={w}
                initial={{ y: "110%" }}
                animate={{ y: 0 }}
                transition={{ duration: 0.9, delay: 0.6 + i * 0.1, ease: [0.22, 1, 0.36, 1] }}
                className="text-gradient mr-[0.25em] inline-block"
              >
                {w}
              </motion.span>
            ))}
          </span>
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.8 }}
          className="mt-8 max-w-xl text-lg leading-relaxed text-ink-2"
        >
          Dr House takes the case, pulls the chart, and calls the colleague who owns the finding.
          Seven agents. One department. Zero guesswork about who answers what.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.15, duration: 0.8 }}
          className="mt-9 flex flex-wrap items-center gap-3"
        >
          <a
            href="#consult"
            className="group inline-flex items-center gap-2 rounded-full bg-ink px-6 py-3.5 text-sm font-medium text-white transition-transform hover:scale-[1.03] active:scale-95"
          >
            Consult Dr House
            <span className="transition-transform group-hover:translate-x-1">→</span>
          </a>
          <a
            href="#dispatch"
            className="rounded-full border border-ink/15 bg-white/60 px-6 py-3.5 text-sm font-medium transition-colors hover:bg-white"
          >
            Watch a case get routed
          </a>
        </motion.div>

        <div className="mt-16 flex gap-12 sm:gap-20">
          <Counter to={7} label="Agents" />
          <Counter to={6} label="Tools" />
          <Counter to={2} label="Skills" />
        </div>
      </motion.div>

      {/* ECG trace */}
      <motion.div style={{ y: ecgY }} className="pointer-events-none absolute inset-x-0 bottom-0">
        <svg viewBox="0 0 1200 280" preserveAspectRatio="none" className="h-48 w-full">
          <path d={ECG} fill="none" stroke="rgba(0,167,157,0.18)" strokeWidth={2} />
          <motion.path
            d={ECG}
            fill="none"
            stroke="#00a79d"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0, pathOffset: 0 }}
            animate={{ pathLength: [0, 0.35, 0.35, 0], pathOffset: [0, 0, 0.65, 1] }}
            transition={{ duration: 4.5, repeat: Infinity, ease: "linear", delay: 1 }}
            style={{ filter: "drop-shadow(0 0 8px rgba(0,167,157,0.8))" }}
          />
        </svg>
      </motion.div>

      <motion.div
        style={{ opacity: fade }}
        className="absolute bottom-8 left-1/2 flex -translate-x-1/2 flex-col items-center gap-2 font-mono text-[10px] uppercase tracking-[0.3em] text-muted"
      >
        Scroll
        <motion.span
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.6, repeat: Infinity }}
          className="h-8 w-px bg-gradient-to-b from-muted to-transparent"
        />
      </motion.div>
    </section>
  );
}
