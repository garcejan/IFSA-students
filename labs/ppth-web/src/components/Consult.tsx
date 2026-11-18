"use client";

import React, { useState, useEffect, useRef, memo } from "react";
import { motion } from "motion/react";

const STARTERS = [
  "Take the case of patient PPTH-003. What are we looking at?",
  "Patient PPTH-001 had sudden aphasia then a seizure. What is causing it?",
  "Patient PPTH-004 is on warfarin. Can we add amiodarone?",
  "The family of PPTH-002 is refusing an LP and threatening to sue.",
];

/**
 * Isolated mount container.
 * Wrapped in React.memo and using dangerouslySetInnerHTML to prevent React 19
 * from reconciling or wiping out third-party DOM nodes injected by wxoLoader
 * when the parent component re-renders (e.g. on animations or click interactions).
 */
const ChatMount = memo(
  function ChatMount() {
    return (
      <div
        id="root"
        className="h-full"
        dangerouslySetInnerHTML={{ __html: "" }}
        suppressHydrationWarning
      />
    );
  },
  () => true
);

/**
 * EMBED POINT
 * ----------
 * `#root` is an empty, full-size container reserved for the chat.
 * Mount your embedded chat into it (e.g. rootElementID: "root").
 * As soon as it has a child element, the placeholder behind it is hidden by CSS
 * (see `.chat-frame:has(#root > *)` in globals.css).
 */
export default function Consult() {
  const [copied, setCopied] = useState<number | null>(null);
  const chatInstanceRef = useRef<any>(null);

  const copy = (text: string, i: number) => {
    navigator.clipboard?.writeText(text);
    setCopied(i);
    setTimeout(() => setCopied(null), 1400);
  };

  // =========================================================================
  // WATSONX ORCHESTRATE EMBED POINT
  // The embedding script belongs here inside a useEffect hook.
  // This guarantees that #root is rendered in the DOM before
  // wxoLoader initializes.
  // =========================================================================
  useEffect(() => {
    // =========================================================================
    // TODO: PLACE YOUR CREDENTIALS FROM WATSONX ORCHESTRATE HERE
    // Follow the tutorial in /labs/WORKSHOP-part2-EMBED.md to obtain these values.
    // =========================================================================

    const orchestrationID = "";  // PLACE HERE: orchestrationID 
    const hostURL = "";  // PLACE HERE: hostURL
    const crn = "";  // PLACE HERE: crn
    const agentId = "";  // PLACE HERE: agentId
    const agentEnvironmentId = "";  // PLACE HERE: agentEnvironmentId

    const initChat = () => {
      const container = document.getElementById("root");
      if (!container || !(window as any).wxoLoader) return;

      const config = {
        orchestrationID,
        hostURL,
        rootElementID: "root",
        deploymentPlatform: "ibmcloud",
        crn,
        chatOptions: {
          agentId,
          agentEnvironmentId,
          onLoad: (instance: any) => {
            chatInstanceRef.current = instance;
          },
        },
        layout: {
          form: "custom",
          showOrchestrateHeader: true,
          customElement: container,
        },
        showLauncher: false,
      };

      (window as any).wxOConfiguration = config;
      (window as any).wxoLoader.init(config);
    };

    const scriptId = "wxo-loader-script";
    if (!document.getElementById(scriptId)) {
      const script = document.createElement("script");
      script.id = scriptId;

      const cleanHostURL = hostURL.startsWith("/*") ? "https://eu-de.watson-orchestrate.cloud.ibm.com" : hostURL;
      script.src = `${cleanHostURL}/wxochat/wxoLoader.js?embed=true`;
      script.addEventListener("load", () => {
        initChat();
      });
      document.head.appendChild(script);
    } else if ((window as any).wxoLoader) {
      initChat();
    }

    return () => {
      if (chatInstanceRef.current && typeof chatInstanceRef.current.destroy === "function") {
        try {
          chatInstanceRef.current.destroy();
        } catch {
          // Ignore destroy errors during unmount
        }
        chatInstanceRef.current = null;
      }
    };
  }, []);

  return (
    <section id="consult" className="grid-paper-dark relative overflow-hidden bg-night px-6 py-32 text-white">
      <div className="pointer-events-none absolute left-1/2 top-0 h-[500px] w-[900px] -translate-x-1/2 rounded-full bg-[radial-gradient(ellipse,rgba(25,195,230,0.22),transparent_65%)] blur-2xl" />

      <div className="relative mx-auto grid max-w-6xl gap-12 lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.25em] text-cyan">Consult</p>
          <h2 className="mt-3 text-[clamp(2.2rem,4.8vw,4rem)] font-semibold leading-[1] tracking-[-0.035em]">
            Take a case to <span className="text-gradient">Dr House.</span>
          </h2>
          <p className="mt-5 max-w-md text-lg leading-relaxed text-white/60">
            Quote a patient ID and describe the finding. House pulls the chart, then calls whoever owns it.
          </p>

          <p className="mt-10 font-mono text-[11px] uppercase tracking-[0.2em] text-white/40">Try one · click to copy</p>
          <div className="mt-3 flex flex-col gap-2">
            {STARTERS.map((s, i) => (
              <motion.button
                key={s}
                onClick={() => copy(s, i)}
                whileHover={{ x: 6 }}
                whileTap={{ scale: 0.98 }}
                className="glass-dark rounded-2xl px-4 py-3 text-left text-sm text-white/80 transition-colors hover:text-white"
              >
                {s}
                <span className="ml-2 font-mono text-[10px] text-cyan">{copied === i ? "copied ✓" : ""}</span>
              </motion.button>
            ))}
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 70, scale: 0.96 }}
          whileInView={{ opacity: 1, y: 0, scale: 1 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="chat-frame relative h-[680px] overflow-hidden rounded-[32px] border border-white/15 bg-[#0a1a28] shadow-[0_40px_120px_-30px_rgba(25,195,230,0.35)]"
        >
          {/* Placeholder — hidden automatically once the chat is mounted */}
          <div className="chat-placeholder absolute inset-0 flex flex-col">
            <div className="flex items-center gap-3 border-b border-white/10 px-6 py-4">
              <span className="grid h-9 w-9 place-items-center rounded-full bg-gradient-to-br from-cyan to-[#0b4f7a] font-mono text-xs">GH</span>
              <div>
                <div className="text-sm font-semibold">Dr House</div>
                <div className="font-mono text-[10px] uppercase tracking-widest text-white/40">dr_house · orchestrator</div>
              </div>
              <span className="ml-auto flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-white/40">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400" /> Awaiting channel
              </span>
            </div>
            <div className="relative flex flex-1 flex-col items-center justify-center overflow-hidden px-8 text-center">
              <div className="scanline pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-transparent via-cyan/10 to-transparent" />
              <svg viewBox="0 0 200 60" className="mb-6 h-14 w-48">
                <motion.path
                  d="M0 30 H55 L65 30 L72 12 L82 52 L92 30 H120 L128 24 L136 30 H200"
                  fill="none"
                  stroke="#19c3e6"
                  strokeWidth={2.5}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  animate={{ pathLength: [0, 1, 1], opacity: [1, 1, 0] }}
                  transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                />
              </svg>
              <p className="text-xl font-medium">The consultation room is being prepared</p>
              <p className="mt-2 max-w-sm text-sm text-white/45">
                The embedded chat will appear here. This container is{" "}
                <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-xs text-cyan">#root</code>.
              </p>
            </div>
          </div>

          {/* Mount point. Protected from React reconciliation by ChatMount memo and dangerouslySetInnerHTML */}
          <ChatMount />
        </motion.div>
      </div>
    </section>
  );
}
