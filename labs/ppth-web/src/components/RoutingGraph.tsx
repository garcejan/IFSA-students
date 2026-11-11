"use client";

import { motion } from "motion/react";
import { COLLABORATORS, HOUSE, type AgentId } from "@/lib/data";

const CENTER = { x: 400, y: 285 };

const POS: Record<string, { x: number; y: number }> = {
  dr_foreman: { x: 400, y: 80 },
  dr_wilson: { x: 645, y: 175 },
  dr_cameron: { x: 645, y: 395 },
  dr_cuddy: { x: 400, y: 490 },
  dr_chase: { x: 155, y: 395 },
  case_reviewer: { x: 155, y: 175 },
};

type Props = {
  /** The colleague currently being consulted, if any. */
  target: AgentId | null;
  /** 0 idle · 1 case arrives · 2 chart pulled · 3 routed · 4 answer returns */
  phase: number;
  dark?: boolean;
};

export default function RoutingGraph({ target, phase, dark = false }: Props) {
  const stroke = dark ? "rgba(255,255,255,0.14)" : "rgba(11,27,43,0.12)";
  const label = dark ? "#e6f1f7" : "#0b1b2b";
  const sub = dark ? "#7fa0b5" : "#7b8ea0";
  const nodeFill = dark ? "#0c2233" : "#ffffff";
  const targetPos = target ? POS[target] : null;
  const targetColor = target ? COLLABORATORS.find((c) => c.id === target)!.color : "#19c3e6";

  return (
    <svg viewBox="0 0 800 590" className="h-full w-full" role="img" aria-label="Routing graph: Dr House delegating to his colleagues">
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="7" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id="houseGrad" cx="50%" cy="35%" r="70%">
          <stop offset="0%" stopColor="#19c3e6" />
          <stop offset="100%" stopColor="#0b4f7a" />
        </radialGradient>
      </defs>

      {/* idle spokes */}
      {COLLABORATORS.map((c) => (
        <line
          key={c.id}
          x1={CENTER.x}
          y1={CENTER.y}
          x2={POS[c.id].x}
          y2={POS[c.id].y}
          stroke={stroke}
          strokeWidth={1.5}
          strokeDasharray="4 7"
        />
      ))}

      {/* active spoke */}
      {targetPos && (
        <motion.line
          key={`spoke-${target}`}
          x1={CENTER.x}
          y1={CENTER.y}
          x2={targetPos.x}
          y2={targetPos.y}
          stroke={targetColor}
          strokeWidth={3}
          filter="url(#glow)"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: phase >= 3 ? 1 : 0, opacity: phase >= 3 ? 1 : 0 }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
        />
      )}

      {/* inbound case packet */}
      <motion.circle
        r={7}
        fill="#19c3e6"
        filter="url(#glow)"
        initial={false}
        animate={{
          cx: CENTER.x,
          cy: phase === 1 ? CENTER.y : CENTER.y - 255,
          opacity: phase === 1 ? 1 : 0,
        }}
        transition={{ duration: 0.9, ease: "easeOut" }}
      />

      {/* answer packet travelling back */}
      {targetPos && (
        <motion.circle
          key={`pkt-${target}`}
          r={7}
          fill={targetColor}
          filter="url(#glow)"
          initial={false}
          animate={{
            cx: phase === 4 ? CENTER.x : targetPos.x,
            cy: phase === 4 ? CENTER.y : targetPos.y,
            opacity: phase === 4 ? [0, 1, 1, 0] : 0,
          }}
          transition={{ duration: 1.1, ease: "easeInOut" }}
        />
      )}

      {/* chart pulse around House */}
      {phase === 2 && (
        <motion.circle
          cx={CENTER.x}
          cy={CENTER.y}
          fill="none"
          stroke="#19c3e6"
          strokeWidth={2}
          initial={{ r: 50, opacity: 0.9 }}
          animate={{ r: 130, opacity: 0 }}
          transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
        />
      )}

      {/* collaborators */}
      {COLLABORATORS.map((c) => {
        const p = POS[c.id];
        const active = target === c.id && phase >= 3;
        return (
          <g key={c.id}>
            <motion.circle
              cx={p.x}
              cy={p.y}
              fill={nodeFill}
              stroke={active ? c.color : stroke}
              strokeWidth={active ? 3 : 1.5}
              filter={active ? "url(#glow)" : undefined}
              initial={false}
              animate={{ r: active ? 44 : 36 }}
              transition={{ type: "spring", stiffness: 220, damping: 18 }}
            />
            <text
              x={p.x}
              y={p.y + 6}
              textAnchor="middle"
              fontSize={active ? 19 : 16}
              fontWeight={600}
              fill={active ? c.color : label}
              style={{ fontFamily: "var(--font-plex-mono)" }}
            >
              {c.initials}
            </text>
            <text x={p.x} y={p.y + 64} textAnchor="middle" fontSize={15} fontWeight={600} fill={label}>
              {c.short}
            </text>
            <text x={p.x} y={p.y + 82} textAnchor="middle" fontSize={11.5} fill={sub}>
              {c.role.split(" · ")[0]}
            </text>
          </g>
        );
      })}

      {/* House */}
      <motion.circle
        cx={CENTER.x}
        cy={CENTER.y}
        fill="url(#houseGrad)"
        filter="url(#glow)"
        initial={false}
        animate={{ r: phase === 0 ? [52, 56, 52] : 58 }}
        transition={phase === 0 ? { duration: 2.4, repeat: Infinity } : { type: "spring" }}
      />
      <text x={CENTER.x} y={CENTER.y + 8} textAnchor="middle" fontSize={24} fontWeight={600} fill="#fff" style={{ fontFamily: "var(--font-plex-mono)" }}>
        {HOUSE.initials}
      </text>
      <text x={CENTER.x} y={CENTER.y + 88} textAnchor="middle" fontSize={15} fontWeight={600} fill={label}>
        House
      </text>
      <text x={CENTER.x} y={CENTER.y + 105} textAnchor="middle" fontSize={11.5} fill={sub}>
        orchestrator
      </text>
    </svg>
  );
}
