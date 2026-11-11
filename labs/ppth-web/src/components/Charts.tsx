"use client";

import { motion, useMotionValue, useSpring, useTransform } from "motion/react";
import { PATIENTS, byId, type Patient } from "@/lib/data";

function Chart({ p, i }: { p: Patient; i: number }) {
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rx = useSpring(useTransform(my, [-0.5, 0.5], [9, -9]), { stiffness: 200, damping: 20 });
  const ry = useSpring(useTransform(mx, [-0.5, 0.5], [-9, 9]), { stiffness: 200, damping: 20 });
  const owner = byId(p.route);

  return (
    <motion.div
      initial={{ opacity: 0, y: 60 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.8, delay: (i % 2) * 0.12, ease: [0.22, 1, 0.36, 1] }}
      style={{ perspective: 900 }}
    >
      <motion.article
        style={{ rotateX: rx, rotateY: ry, transformStyle: "preserve-3d" }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          mx.set((e.clientX - r.left) / r.width - 0.5);
          my.set((e.clientY - r.top) / r.height - 0.5);
        }}
        onMouseLeave={() => {
          mx.set(0);
          my.set(0);
        }}
        className="relative overflow-hidden rounded-[28px] border border-line bg-white p-7 shadow-[0_24px_60px_-34px_rgba(11,27,43,0.4)]"
      >
        <div className="flex items-start justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted">{p.id} · day {p.day}</p>
            <h3 className="mt-1 text-2xl font-semibold tracking-tight">{p.name}</h3>
            <p className="text-sm text-ink-2">{p.age} · {p.sex === "F" ? "Female" : "Male"}</p>
          </div>
          <span
            className="rounded-full px-3 py-1 font-mono text-[11px] text-white"
            style={{ background: owner.color }}
          >
            → {owner.short}
          </span>
        </div>

        <p className="mt-5 text-[16px] leading-snug">{p.complaint}</p>

        <div className="mt-5 grid grid-cols-4 gap-2">
          {p.vitals.map((v) => (
            <div key={v.label} className={`rounded-2xl p-3 ${v.flag ? "bg-signal/10" : "bg-paper"}`}>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted">{v.label}</div>
              <div className={`mt-0.5 text-[13px] font-semibold ${v.flag ? "text-signal" : ""}`}>{v.value}</div>
            </div>
          ))}
        </div>

        <p className="mt-5 border-t border-line pt-4 text-sm text-ink-2">{p.history}</p>
      </motion.article>
    </motion.div>
  );
}

export default function Charts() {
  return (
    <section id="charts" className="grid-paper relative px-6 py-32">
      <div className="mx-auto max-w-6xl">
        <p className="font-mono text-xs uppercase tracking-[0.25em] text-teal">On the ward</p>
        <h2 className="mt-3 max-w-3xl text-[clamp(2rem,4.4vw,3.6rem)] font-semibold leading-[1.02] tracking-[-0.03em]">
          Four charts. Four different doors.
        </h2>
        <p className="mt-4 max-w-xl text-lg text-ink-2">
          Synthetic patients, real routing. Quote an ID in the chat and House reads the same chart you see here.
        </p>
        <div className="mt-14 grid gap-6 md:grid-cols-2">
          {PATIENTS.map((p, i) => (
            <Chart key={p.id} p={p} i={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
