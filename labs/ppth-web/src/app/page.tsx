import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import Manifesto from "@/components/Manifesto";
import Dispatch from "@/components/Dispatch";
import Team from "@/components/Team";
import RouterLab from "@/components/RouterLab";
import Charts from "@/components/Charts";
import Toolbelt from "@/components/Toolbelt";
import Consult from "@/components/Consult";

export default function Home() {
  return (
    <main>
      <Nav />
      <Hero />
      <Manifesto />
      <Dispatch />
      <Team />
      <RouterLab />
      <Charts />
      <Toolbelt />
      <Consult />
      <footer className="bg-night px-6 pb-12 pt-8 text-center font-mono text-xs text-white/40">
        <div className="mx-auto max-w-6xl border-t border-white/10 pt-8">
          Princeton-Plainsboro Teaching Hospital · Department of Diagnostic Medicine
          <br />
          Synthetic teaching case — not medical advice.
        </div>
      </footer>
    </main>
  );
}
