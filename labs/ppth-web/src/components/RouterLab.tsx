"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import RoutingGraph from "./RoutingGraph";
import { SCENARIOS, byId, type Scenario } from "@/lib/data";

export default function RouterLab() {
  const [active, setActive] = useState<Scenario | null>(null);
  const [phase, setPhase] = useState(0);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clear = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  useEffect(() => clear, []);

  const run = (s: Scenario) => {
    clear();
    setActive(s);
    setPhase(1);
    timers.current = [
      setTimeout(() => setPhase(2), 900),
      setTimeout(() => setPhase(3), 2100),
      setTimeout(() => setPhase(4), 3300),
    ];
  };

  const target = active ? byId(active.target) : null;

  return (
    <section id="router" className="relative bg-white px-6 py-32">
      <div className="mx-auto max-w-6xl">
        <p className="font-mono text-xs uppercase tracking-[0.25em] text-teal">Routing lab</p>
        <h2 className="mt-3 max-w-3xl text-[clamp(2rem,4.4vw,3.6rem)] font-semibold leading-[1.02] tracking-[-0.03em]">
          Pick a case. Watch who answers.
        </h2>
        <p className="mt-4 max-w-xl text-lg text-ink-2">
          An illustration of House’s routing rules. The real thing happens in the chat below.
        </p>

        <div className="mt-14 grid items-start gap-10 lg:grid-cols-[1fr_1.05fr]">
          <div className="flex flex-col gap-3">
            {SCENARIOS.map((s) => {
              const sel = active?.prompt === s.prompt;
              return (
                <motion.button
                  key={s.prompt}
                  onClick={() => run(s)}
                  whileHover={{ x: 6 }}
                  whileTap={{ scale: 0.98 }}
                  className={`rounded-2xl border p-5 text-left transition-colors ${
                    sel ? "border-ink bg-ink text-white" : "border-line bg-paper hover:border-ink/30"
                  }`}
                >
                  <span className={`font-mono text-[11px] uppercase tracking-[0.18em] ${sel ? "text-cyan" : "text-muted"}`}>
                    Case
                  </span>
                  <span className="mt-1.5 block text-[15.5px] leading-snug">{s.prompt}</span>
                </motion.button>
              );
            })}
          </div>

          <div className="lg:sticky lg:top-24">
            <div className="relative overflow-hidden rounded-[32px] bg-night p-4 text-white">
              <div className="grid-paper-dark pointer-events-none absolute inset-0" />
              <div className="relative aspect-[800/590] w-full">
                <RoutingGraph target={active ? active.target : null} phase={phase} dark />
              </div>
            </div>

            <div className="mt-4 min-h-[120px] rounded-3xl border border-line bg-paper p-6">
              <AnimatePresence mode="wait">
                {active && target && phase >= 3 ? (
                  <motion.div
                    key={active.prompt}
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-full px-3 py-1 font-mono text-xs text-white" style={{ background: target.color }}>
                        → {target.id}
                      </span>
                      <span className="font-mono text-xs text-muted">{active.tool}</span>
                    </div>
                    <p className="mt-3 text-[15.5px] leading-relaxed text-ink-2">{active.why}</p>
                  </motion.div>
                ) : (
                  <motion.p key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-ink-2">
                    {active ? "House is working the case…" : "Select a case on the left."}
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
