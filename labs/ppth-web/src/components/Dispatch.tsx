"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion, useMotionValueEvent, useScroll, useTransform } from "motion/react";
import RoutingGraph from "./RoutingGraph";

const STEPS = [
  {
    n: "01",
    title: "A case walks in",
    body: "“Patient PPTH-001 had sudden aphasia, then a seizure. What is causing it?” One question, one entry point: Dr House.",
    chip: "user → dr_house",
  },
  {
    n: "02",
    title: "House pulls the chart",
    body: "He never invents a symptom. Before any theory, get_patient_record reads age, vitals, medication and history straight from the chart.",
    chip: "tool · get_patient_record",
  },
  {
    n: "03",
    title: "He calls the owner of the finding",
    body: "Aphasia and seizure are neurological. House could reason about them himself. The protocol says he must not. The case goes to Dr Foreman.",
    chip: "delegate → dr_foreman",
  },
  {
    n: "04",
    title: "The answer comes back",
    body: "Foreman runs differential_diagnosis, leads with red flags, and names one confirming test per candidate. House presents it, naming who he consulted.",
    chip: "tool · differential_diagnosis",
  },
];

export default function Dispatch() {
  const ref = useRef<HTMLElement>(null);
  const [step, setStep] = useState(0);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const bar = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  useMotionValueEvent(scrollYProgress, "change", (p) => {
    setStep(Math.min(STEPS.length - 1, Math.max(0, Math.floor(p * STEPS.length))));
  });

  const phase = step + 1;

  return (
    <section id="dispatch" ref={ref} className="grid-paper-dark relative h-[420vh] bg-night text-white">
      <div className="sticky top-0 flex h-screen items-center overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_70%_50%,rgba(25,195,230,0.14),transparent_60%)]" />
        <div className="relative mx-auto grid w-full max-w-6xl items-center gap-6 px-6 pt-20 lg:grid-cols-[0.9fr_1.1fr]">
          <div>
            <p className="mb-5 font-mono text-xs uppercase tracking-[0.25em] text-cyan">Delegation, step by step</p>

            <div className="relative min-h-[270px]">
              <AnimatePresence mode="wait">
                <motion.div
                  key={step}
                  initial={{ opacity: 0, y: 28, filter: "blur(8px)" }}
                  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                  exit={{ opacity: 0, y: -24, filter: "blur(8px)" }}
                  transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                >
                  <div className="font-mono text-6xl font-light text-cyan/70">{STEPS[step].n}</div>
                  <h2 className="mt-3 text-[clamp(1.9rem,3.6vw,3.1rem)] font-semibold leading-[1.05] tracking-[-0.03em]">
                    {STEPS[step].title}
                  </h2>
                  <p className="mt-5 max-w-md text-[17px] leading-relaxed text-white/65">{STEPS[step].body}</p>
                  <span className="glass-dark mt-6 inline-block rounded-full px-4 py-1.5 font-mono text-xs text-cyan">
                    {STEPS[step].chip}
                  </span>
                </motion.div>
              </AnimatePresence>
            </div>

            <div className="mt-8 flex items-center gap-3">
              {STEPS.map((s, i) => (
                <span
                  key={s.n}
                  className={`h-1.5 rounded-full transition-all duration-500 ${i === step ? "w-10 bg-cyan" : "w-4 bg-white/20"}`}
                />
              ))}
            </div>
          </div>

          <div className="relative mx-auto aspect-[800/590] w-full max-w-[640px]">
            <RoutingGraph target="dr_foreman" phase={phase} dark />
          </div>
        </div>

        <div className="absolute inset-x-0 bottom-0 h-[3px] bg-white/10">
          <motion.div style={{ width: bar }} className="h-full bg-gradient-to-r from-teal to-cyan" />
        </div>
      </div>
    </section>
  );
}
