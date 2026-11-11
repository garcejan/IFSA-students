"use client";

import { motion } from "motion/react";
import { SKILLS, TOOLS } from "@/lib/data";

const container = { hidden: {}, show: { transition: { staggerChildren: 0.08 } } };
const item = {
  hidden: { opacity: 0, y: 30, scale: 0.97 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] as const } },
};

export default function Toolbelt() {
  return (
    <section id="toolbelt" className="relative bg-white px-6 py-32">
      <div className="mx-auto max-w-6xl">
        <p className="font-mono text-xs uppercase tracking-[0.25em] text-teal">Instruments</p>
        <h2 className="mt-3 max-w-3xl text-[clamp(2rem,4.4vw,3.6rem)] font-semibold leading-[1.02] tracking-[-0.03em]">
          Six tools. Two skills. No answers from memory.
        </h2>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-100px" }}
          className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
        >
          {TOOLS.map((t) => (
            <motion.div
              key={t.name}
              variants={item}
              whileHover={{ y: -6 }}
              className="group relative overflow-hidden rounded-3xl border border-line bg-paper p-6"
            >
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">Tool</p>
              <h3 className="mt-2 font-mono text-[17px] font-medium">{t.name}</h3>
              <p className="mt-3 text-[15px] leading-snug text-ink-2">{t.blurb}</p>
              <p className="mt-4 font-mono text-[11px] text-teal">{t.owner}</p>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-100px" }}
          className="mt-4 grid gap-4 md:grid-cols-2"
        >
          {SKILLS.map((s) => (
            <motion.div key={s.name} variants={item} className="relative overflow-hidden rounded-3xl bg-ink p-7 text-white">
              <div className="pointer-events-none absolute -right-10 -top-10 h-44 w-44 rounded-full bg-cyan/25 blur-3xl" />
              <p className="relative font-mono text-[11px] uppercase tracking-[0.18em] text-cyan">Agent skill</p>
              <h3 className="relative mt-2 font-mono text-xl font-medium">{s.name}</h3>
              <p className="relative mt-3 max-w-md text-[15px] leading-snug text-white/65">{s.blurb}</p>
              <p className="relative mt-4 font-mono text-[11px] text-white/40">{s.owner}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
