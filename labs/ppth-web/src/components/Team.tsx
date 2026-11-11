"use client";

import { useRef, useState, useLayoutEffect } from "react";
import { motion, useScroll, useTransform } from "motion/react";
import { COLLABORATORS, HOUSE, type Agent } from "@/lib/data";

function Card({ a, i }: { a: Agent; i: number }) {
  return (
    <article className="group relative flex h-[min(560px,72vh)] w-[min(380px,82vw)] shrink-0 flex-col justify-between overflow-hidden rounded-[32px] border border-white/70 bg-white p-8 shadow-[0_30px_80px_-40px_rgba(11,27,43,0.45)]">
      <div
        className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full opacity-25 blur-3xl transition-opacity duration-500 group-hover:opacity-60"
        style={{ background: a.color }}
      />
      <div className="relative flex items-start justify-between">
        <div
          className="grid h-20 w-20 place-items-center rounded-3xl font-mono text-2xl font-medium text-white"
          style={{ background: `linear-gradient(135deg, ${a.color}, ${a.color}aa)` }}
        >
          {a.initials}
        </div>
        <span className="font-mono text-xs text-muted">{String(i + 1).padStart(2, "0")}</span>
      </div>

      <div className="relative">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em]" style={{ color: a.color }}>
          {a.role}
        </p>
        <h3 className="mt-2 text-3xl font-semibold tracking-tight">{a.name}</h3>
        <p className="mt-4 text-[17px] italic leading-snug text-ink-2">“{a.quote}”</p>
      </div>

      <div className="relative">
        <div className="flex flex-wrap gap-1.5">
          {a.owns.map((o) => (
            <span key={o} className="rounded-full bg-paper px-3 py-1 text-xs text-ink-2">
              {o}
            </span>
          ))}
        </div>
        <div className="mt-4 border-t border-line pt-3 font-mono text-[11px] text-muted">
          {a.id}
          {a.tools.length > 0 && <span> · {a.tools.join(" · ")}</span>}
        </div>
      </div>
    </article>
  );
}

export default function Team() {
  const ref = useRef<HTMLElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const [dist, setDist] = useState(0);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const x = useTransform(scrollYProgress, [0, 1], [0, -dist]);

  useLayoutEffect(() => {
    const el = track.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setDist(Math.max(0, el.scrollWidth - window.innerWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <section id="team" ref={ref} className="relative h-[360vh] bg-paper">
      <div className="sticky top-0 flex h-screen flex-col justify-center overflow-hidden">
        <div className="mx-auto mb-10 w-full max-w-6xl px-6 pt-16">
          <p className="font-mono text-xs uppercase tracking-[0.25em] text-teal">The department</p>
          <h2 className="mt-3 max-w-2xl text-[clamp(2rem,4.4vw,3.6rem)] font-semibold leading-[1.02] tracking-[-0.03em]">
            One orchestrator. Five specialists. One observer.
          </h2>
        </div>

        <motion.div ref={track} style={{ x }} className="flex w-max items-center gap-6 px-6 lg:pl-[max(1.5rem,calc((100vw-72rem)/2+1.5rem))] pr-24">
          {/* House — dark hero card */}
          <article className="relative flex h-[min(560px,72vh)] w-[min(420px,86vw)] shrink-0 flex-col justify-between overflow-hidden rounded-[32px] bg-ink p-8 text-white">
            <div className="grid-paper-dark pointer-events-none absolute inset-0 opacity-70" />
            <div className="pointer-events-none absolute -bottom-24 -left-16 h-72 w-72 rounded-full bg-cyan/30 blur-3xl" />
            <div className="relative grid h-20 w-20 place-items-center rounded-3xl bg-gradient-to-br from-cyan to-[#0b4f7a] font-mono text-2xl font-medium">
              {HOUSE.initials}
            </div>
            <div className="relative">
              <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-cyan">{HOUSE.role}</p>
              <h3 className="mt-2 text-4xl font-semibold tracking-tight">{HOUSE.name}</h3>
              <p className="mt-4 text-lg italic text-white/70">“{HOUSE.quote}”</p>
            </div>
            <div className="relative">
              <div className="flex flex-wrap gap-1.5">
                {HOUSE.owns.map((o) => (
                  <span key={o} className="glass-dark rounded-full px-3 py-1 text-xs text-white/80">
                    {o}
                  </span>
                ))}
              </div>
              <div className="mt-4 border-t border-white/10 pt-3 font-mono text-[11px] text-white/45">
                {HOUSE.id} · {HOUSE.tools.join(" · ")}
              </div>
            </div>
          </article>

          {COLLABORATORS.map((a, i) => (
            <Card key={a.id} a={a} i={i + 1} />
          ))}
        </motion.div>
      </div>
    </section>
  );
}
