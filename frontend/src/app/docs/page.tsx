"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Pacifico } from "next/font/google";

const cursiveFont = Pacifico({
  weight: "400",
  subsets: ["latin"],
  display: "swap",
});

export default function AgastyaDocsPage() {
  const [activeSection, setActiveSection] = useState("overview");

  const sections = [
    { id: "overview", label: "System Overview" },
    { id: "tokenizer", label: "BPE Tokenizer Pipeline" },
    { id: "architecture", label: "12-Layer Transformer" },
    { id: "optimization", label: "Training & Optimization" },
    { id: "deployment", label: "Distributed Free Cloud" },
    { id: "alignment", label: "The Alignment Rule" },
  ];

  const handleScroll = (id: string) => {
    setActiveSection(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="min-h-screen bg-black text-zinc-300 font-sans antialiased selection:bg-[#d2ff00]/30">
      
      {/* GLOBAL TOP NAVIGATION */}
      <header className="w-full border-b border-zinc-950 bg-black/80 backdrop-blur sticky top-0 z-50 px-8 py-4 flex items-center justify-between">
        <Link href="/" className={`${cursiveFont.className} text-xl tracking-wide text-white normal-case`}>
          Agastya Labs
        </Link>
        <div className="flex items-center gap-6">
          <Link href="/" className="text-[10px] font-mono tracking-[0.2em] text-zinc-400 uppercase hover:text-white transition-colors">
            ← Back to Terminal
          </Link>
          <span className="text-[10px] font-mono tracking-[0.2em] text-[#d2ff00] uppercase bg-zinc-950 px-3 py-1 border border-zinc-900 rounded">
            DOCS v1.0.0
          </span>
        </div>
      </header>

      <div className="max-w-[1600px] mx-auto flex">
        
        {/* STICKY DESKTOP SIDEBAR */}
        <aside className="hidden lg:block w-72 h-[calc(screen-64px)] sticky top-16 border-r border-zinc-950 p-8 space-y-8 self-start">
          <div className="space-y-2">
            <div className="text-[10px] font-mono tracking-widest text-zinc-600 uppercase">// CORE CORE ARCHITECTURE</div>
            <nav className="flex flex-col gap-1.5">
              {sections.map((sec) => (
                <button
                  key={sec.id}
                  onClick={() => handleScroll(sec.id)}
                  className={`text-left text-xs font-mono py-2 px-3 rounded-lg border transition-all ${
                    activeSection === sec.id
                      ? "bg-zinc-950 text-[#d2ff00] border-zinc-900 font-bold"
                      : "text-zinc-400 border-transparent hover:text-white hover:bg-zinc-950/40"
                  }`}
                >
                  {sec.label}
                </button>
              ))}
            </nav>
          </div>
        </aside>

        {/* MAIN DOCUMENTATION COMPONENT PANEL */}
        <main className="flex-1 max-w-4xl px-6 py-12 md:p-16 space-y-20 border-r border-zinc-950">
          
          {/* SECTION 1: OVERVIEW */}
          <section id="overview" className="space-y-4 scroll-mt-24">
            <div className="font-mono text-[10px] tracking-widest text-[#d2ff00] uppercase">// PHASE 01</div>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight uppercase">System Architecture Overview</h1>
            <p className="text-zinc-400 text-sm leading-relaxed">
              Project Agastya is a custom-architected 38-million parameter autoregressive language engine engineered entirely out of native PyTorch matrix layers. Designed by Dinesh, this model bypasses pre-packaged libraries to implement a multi-layered causal transformer topology optimized for maximum local stream throughput and ultra-low VRAM profiles.
            </p>
            <div className="bg-zinc-950 border border-zinc-900 rounded-xl p-4 font-mono text-xs text-zinc-400">
              <span className="text-zinc-600 block">// Active Footprint Profile</span>
              Total Computational Weights: <span className="text-white font-bold">38,154,240</span><br />
              Active System Burden Memory: <span className="text-[#d2ff00] font-bold">~184.80 MB</span>
            </div>
          </section>

          {/* SECTION 2: TOKENIZER */}
          <section id="tokenizer" className="space-y-4 scroll-mt-24">
            <div className="font-mono text-[10px] tracking-widest text-[#d2ff00] uppercase">// PHASE 02</div>
            <h2 className="text-2xl font-bold text-white tracking-tight uppercase">Byte-Level BPE Tokenizer</h2>
            <p className="text-zinc-400 text-sm leading-relaxed">
              To handle programming context windows efficiently, a custom Byte-Level Byte Pair Encoding (BPE) pipeline was compiled using <code className="text-zinc-200 bg-zinc-950 px-1.5 py-0.5 rounded text-xs font-mono border border-zinc-900">train_tokenizer.py</code>. 
            </p>
            <p className="text-zinc-400 text-sm leading-relaxed">
              By evaluating character configurations iteratively across a combined training dataset (<code className="text-zinc-300">input.txt</code> + <code className="text-zinc-300">large_input.txt</code>), the tokenizer compresses complex string allocations into a tight, optimized directory of exactly <span className="text-white font-bold">2,000 sub-word tokens</span>, avoiding infinite lookup loops and memory bottlenecks.
            </p>
          </section>

          {/* SECTION 3: TRANSFORMER SPECIFICATIONS */}
          <section id="architecture" className="space-y-4 scroll-mt-24">
            <div className="font-mono text-[10px] tracking-widest text-[#d2ff00] uppercase">// PHASE 03</div>
            <h2 className="text-2xl font-bold text-white tracking-tight uppercase">12-Layer Mathematical Block Core</h2>
            <p className="text-zinc-400 text-sm leading-relaxed">
              Agastya calculates dynamic sub-space alignments using 12 deep transformer block steps. Input embeddings are split across 8 parallel attention heads using an explicit scaling index calculation:
            </p>
            <div className="bg-zinc-950 border border-zinc-900 rounded-xl p-5 font-mono text-xs text-center text-white select-all">
              Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) + IL ) @ V
            </div>
            <p className="text-zinc-400 text-sm leading-relaxed">
              Where <code className="text-zinc-300">IL</code> represents a lower-triangular causal attention mask filled with negative infinity values across its upper segments. This mask blocks the model from looking ahead at future answer chains during autoregressive generation passes.
            </p>
          </section>

          {/* SECTION 4: OPTIMIZATION */}
          <section id="optimization" className="space-y-4 scroll-mt-24">
            <div className="font-mono text-[10px] tracking-widest text-[#d2ff00] uppercase">// PHASE 04</div>
            <h2 className="text-2xl font-bold text-white tracking-tight uppercase">Training & Optimization Loop</h2>
            <p className="text-zinc-400 text-sm leading-relaxed">
              Optimization steps were managed using an AdamW gradient calculator tracking cross-entropy classification losses. The model calculates forward steps on an embedding plane width of 512 dimensions, expanding internally through feed-forward scaling blocks up to 2048 layers before passing through residual add skip states.
            </p>
          </section>

          {/* SECTION 5: DEPLOYMENT */}
          <section id="deployment" className="space-y-4 scroll-mt-24">
            <div className="font-mono text-[10px] tracking-widest text-[#d2ff00] uppercase">// PHASE 05</div>
            <h2 className="text-2xl font-bold text-white tracking-tight uppercase">Distributed Free Cloud Architecture</h2>
            <p className="text-zinc-400 text-sm leading-relaxed">
              The model operates on a completely cost-free production pipeline split across two separate global clouds:
            </p>
            <ul className="space-y-3 font-mono text-xs text-zinc-400 pl-4 border-l-2 border-zinc-900">
              <li>
                <strong className="text-white block">1. UI Presentation Layer (Vercel)</strong>
                Hosts the Next.js web build. Uses asynchronous chunk readers to listen for real-time network fragments.
              </li>
              <li>
                <strong className="text-white block">2. Computation Core (Hugging Face Spaces)</strong>
                Runs a custom Docker-packaged FastAPI app container. Pulls the 38M weights dynamically into its free CPU system memory allocation upon boot.
              </li>
            </ul>
          </section>

          {/* SECTION 6: ALIGNMENT */}
          <section id="alignment" className="space-y-4 scroll-mt-24">
            <div className="font-mono text-[10px] tracking-widest text-[#d2ff00] uppercase">// THE SYSTEM LAW</div>
            <h2 className="text-2xl font-bold text-[#d2ff00] tracking-tight uppercase">The Token Alignment MLOps Rule</h2>
            <p className="text-zinc-400 text-sm leading-relaxed">
              Because an LLM maps language through strict structural integer sequences, changing the training dataset shuffling rules updates your dictionary indexing properties. 
            </p>
            <div className="bg-red-950/20 border border-red-900/40 rounded-xl p-5 text-zinc-300 text-xs leading-relaxed font-mono">
              <span className="text-red-500 font-bold uppercase block mb-1">⚠️ CRITICAL COMPLIANCE NOTICE</span>
              If you re-train your tokenizer, you MUST immediately execute a local fine-tuning training pass to sync the model weights file (<code className="text-white">.pth</code>). Upload both matched assets simultaneously via your deployment controller script, and execute a <span className="text-white underline">Factory Restart</span> on your Hugging Face Space to flush active server RAM cache systems. Failing to do this results in translation corruption (gibberish streams).
            </div>
          </section>

        </main>
      </div>
    </div>
  );
}