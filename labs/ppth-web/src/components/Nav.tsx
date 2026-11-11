"use client";

import { motion, useScroll, useSpring } from "motion/react";

const LINKS = [
  { href: "#manifesto", label: "Mission" },
  { href: "#dispatch", label: "A case walks in" },
  { href: "#team", label: "The team" },
  { href: "#router", label: "Routing lab" },
  { href: "#charts", label: "Charts" },
  { href: "#toolbelt", label: "Toolbelt" },
];

export default function Nav() {
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 28 });

  return (
    <motion.header
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      className="fixed inset-x-0 top-0 z-50 px-4 pt-4"
    >
      <nav className="glass relative mx-auto flex max-w-6xl items-center justify-between overflow-hidden rounded-full px-5 py-2.5 shadow-[0_8px_40px_-12px_rgba(11,27,43,0.25)]">
        <a href="#top" className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-full bg-ink text-white">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round">
              <path d="M12 4v16M4 12h16" />
            </svg>
          </span>
          <span className="text-sm font-semibold leading-tight tracking-tight">
            Princeton-Plainsboro
            <span className="block font-mono text-[10px] font-normal uppercase tracking-[0.18em] text-muted">
              Diagnostic Medicine
            </span>
          </span>
        </a>

        <ul className="hidden items-center gap-1 lg:flex">
          {LINKS.map((l) => (
            <li key={l.href}>
              <a
                href={l.href}
                className="rounded-full px-3.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:bg-ink/5 hover:text-ink"
              >
                {l.label}
              </a>
            </li>
          ))}
        </ul>

        <a
          href="#consult"
          className="rounded-full bg-ink px-4 py-2 text-[13px] font-medium text-white transition-transform hover:scale-[1.04] active:scale-95"
        >
          Open a case
        </a>

        <motion.span
          style={{ scaleX: progress }}
          className="absolute inset-x-0 bottom-0 h-[2px] origin-left bg-gradient-to-r from-teal via-cyan to-[#2a6df4]"
        />
      </nav>
    </motion.header>
  );
}
