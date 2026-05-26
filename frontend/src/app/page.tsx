"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { Pacifico } from "next/font/google";

// Initialize the brand cursive font natively for Next.js
const cursiveFont = Pacifico({
  weight: "400",
  subsets: ["latin"],
  display: "swap",
});

export default function AgastyaFullConnectedPage() {
  const [inputQuery, setInputQuery] = useState("");
  const [terminalLines, setTerminalLines] = useState([
    { type: "user", text: "What is a pointer in programming?" },
    { type: "ai", text: "A pointer is a variable that holds the memory address of another variable rather than holding a direct value natively." }
  ]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [apiStatus] = useState("ONLINE");

  // Reference hooks to bind directly to the terminal scrolling viewport
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const specs = [
    { label: "TOTAL PARAMETERS", value: "38,154,240", sub: "Optimized Weight Matrices" },
    { label: "CONTEXT HORIZON", value: "256 Tokens", sub: "Attention Memory Span" },
    { label: "ATTENTION LAYERS", value: "12 Blocks", sub: "Causal Transformer Depth" },
    { label: "ATTENTION HEADS", value: "8 Heads", sub: "Parallel Context Windows" },
    { label: "HIDDEN CHANNELS", value: "512 Dims", sub: "Internal Vector Width" },
    { label: "VOCAB DICTIONARY", value: "2,000 Tokens", sub: "Byte-Level BPE Density" }
  ];

  // 🎯 REAL-TIME SCROLL ANCHOR & BROWSER TAB TITLE OVERRIDE
  useEffect(() => {
    // Overrides the generic "Create Next App" template title instantly
    document.title = "Agastya AI | Live Workspace Matrix";

    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [terminalLines]);

  // 📡 CLOUD API STREAM CONNECTION PIPELINE
  const handleExecute = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isStreaming) return;

    const userPrompt = inputQuery;
    setTerminalLines((prev) => [...prev, { type: "user", text: userPrompt }]);
    setInputQuery("");
    setIsStreaming(true);

    // Seed empty placeholder row to receive real-time chunk fragments
    setTerminalLines((prev) => [...prev, { type: "ai", text: "" }]);

    try {
      // Establish live stream reader handshake straight to your Hugging Face Space backend container
      const response = await fetch("https://dinesh05976-agastya-ai.hf.space/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userPrompt }),
      });

      if (!response.ok) throw new Error("Server handshake rejected context parsing execution rules.");

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let compiledResponseString = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const standardTextChunk = decoder.decode(value, { stream: true });
          compiledResponseString += standardTextChunk;

          setTerminalLines((prev) => {
            const architecturalCopy = [...prev];
            architecturalCopy[architecturalCopy.length - 1] = { 
              type: "ai", 
              text: compiledResponseString 
            };
            return architecturalCopy;
          });
        }
      }
    } catch (error) {
      console.error("[CONNECTION FATAL]", error);
      setTerminalLines((prev) => {
        const architecturalCopy = [...prev];
        architecturalCopy[architecturalCopy.length - 1] = { 
          type: "ai", 
          text: "⚠️ CRITICAL HANDSHAKE ERROR: The client UI could not bind to your Hugging Face Space endpoint. Ensure your container instance status is active and running." 
        };
        return architecturalCopy;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const scrollToWorkspace = () => {
    document.getElementById("workspace")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-black text-white selection:bg-[#d2ff00]/30 scroll-smooth font-sans antialiased">
      
      {/* SECTION 1: HERO FOLD */}
      <section className="min-h-screen w-full relative flex flex-col justify-between overflow-hidden border-b border-zinc-900 select-none pb-24 md:pb-0">
        
        {/* RESPONSIVE NAVBAR LAYOUT */}
        <header className="w-full max-w-[1600px] mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between gap-6 z-30">
          <Link href="/" className={`${cursiveFont.className} text-xl tracking-wide text-zinc-300 normal-case`}>
            Agastya Labs
          </Link>

          {/* CONNECTED NAVIGATION MATRIX LINKS - FULLY VISIBLE ACROSS ALL PLATFORMS */}
          <nav className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[9px] sm:text-[10px] font-mono tracking-[0.25em] text-zinc-400 uppercase">
            <span onClick={scrollToWorkspace} className="hover:text-white cursor-pointer transition-colors">Workspace</span>
            <a href="#matrix-specs" className="hover:text-white transition-colors">Architecture</a>
            <Link href="/docs" className="hover:text-white text-[#d2ff00] transition-colors cursor-pointer">Documentation</Link>
            <Link href="/article" className="hover:text-white transition-colors cursor-pointer text-zinc-400">Article</Link>
          </nav>

          <div className="flex items-center gap-6">
            <span className="text-[10px] font-mono tracking-[0.2em] text-[#d2ff00] uppercase hidden lg:inline">
              SYS STATUS: {apiStatus}
            </span>
            <button 
              onClick={scrollToWorkspace} 
              className="bg-[#d2ff00] text-black font-mono font-bold uppercase text-[10px] tracking-[0.15em] rounded-full px-6 py-2.5 hover:bg-[#bcdc00] transition-all transform active:scale-95 shadow-lg shadow-[#d2ff00]/15"
            >
              Get Started
            </button>
          </div>
        </header>

        {/* SUBTLE BACKGROUND GRID MESH PATTERN */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#111_1px,transparent_1px),linear-gradient(to_bottom,#111_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_40%,#000_70%,transparent_100%)] opacity-30 pointer-events-none z-0" />

        {/* ABSTRACT GRADIENT FIELD CORE */}
        <div className="absolute top-[35%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[250px] bg-gradient-to-tr from-zinc-700 via-zinc-900 to-black rounded-full blur-[120px] opacity-20 pointer-events-none z-0" />

        {/* HERO TYPOGRAPHY BOUNDARY */}
        <div className="w-full absolute bottom-0 left-0 right-0 z-10 overflow-hidden translate-y-[25%] pointer-events-none hidden md:block">
          <h1 className="text-[18vw] font-black uppercase tracking-tighter leading-none text-white text-center transform scale-y-105 select-none">
            Agastya
          </h1>
        </div>

        {/* MOBILE FALLBACK HERO TITLE */}
        <div className="md:hidden flex-1 flex items-center justify-center px-6 text-center z-10 relative">
          <h1 className="text-6xl font-black uppercase tracking-tighter text-white">
            AGASTYA
          </h1>
        </div>

        {/* SCROLL BUTTON */}
        <div className="absolute bottom-8 right-8 z-20 flex flex-col items-center gap-2 animate-bounce">
          <button 
            onClick={scrollToWorkspace}
            className="h-10 w-10 rounded-full border border-zinc-800 bg-zinc-950 flex items-center justify-center text-zinc-400 hover:text-white hover:border-zinc-600 transition-all shadow-xl"
            aria-label="Scroll Down"
          >
            ↓
          </button>
        </div>
      </section>

      {/* SECTION 2: WORKSPACE AREA */}
      <section id="workspace" className="max-w-[1400px] mx-auto px-6 py-24 grid grid-cols-1 lg:grid-cols-12 gap-16 items-start relative z-20">
        
        {/* LEFT COLUMN INFO */}
        <div className="lg:col-span-4 space-y-6 lg:sticky lg:top-28">
          <div className="font-mono text-[10px] tracking-widest text-[#d2ff00] uppercase">
            // LIVE INFERENCE WORKSPACE
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white uppercase sm:text-4xl">
            Stream Language Parameters Natively
          </h2>
          <p className="text-zinc-400 text-sm leading-relaxed max-w-md">
            Interact with the freshly calibrated 38M parameter weight layout. Sub-word token vectors are processed on cloud CPU container matrices with integrated stop-guards to prevent infinite loop generations.
          </p>
          <div className="pt-2 space-y-2">
            <div>
              <span className="text-[11px] font-mono text-zinc-600 uppercase block">Engine Architecture Precision:</span>
              <span className="text-xs font-mono text-zinc-300 mt-1 block">32-bit Floating-Point (FP32) Core</span>
            </div>
            <Link href="/article" className="text-xs font-mono text-[#d2ff00] hover:underline block pt-2">
              [ Read Full Technical Article → ]
            </Link>
          </div>
        </div>

        {/* RIGHT COLUMN: BRUTALIST TERMINAL */}
        <div className="lg:col-span-8 w-full rounded-2xl border border-zinc-800 bg-zinc-950 p-6 sm:p-8 flex flex-col justify-between min-h-[460px] shadow-2xl relative">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-900 pb-4 mb-6">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-800 block" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-800 block" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-800 block" />
              <span className="text-[11px] font-mono text-zinc-500 ml-2">agastya_inference_loop.sh</span>
            </div>
            <span className="text-[9px] font-mono text-zinc-500 bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded self-start sm:self-auto">
              CTX_HORIZON=256
            </span>
          </div>

          {/* Message Stack Area */}
          <div 
            ref={scrollContainerRef}
            className="flex-1 space-y-6 overflow-y-auto max-h-[320px] pr-2 text-xs sm:text-sm font-mono leading-relaxed scroll-smooth"
          >
            {terminalLines.map((line, i) => (
              <div key={i} className="space-y-1.5">
                <div className={`text-[10px] font-bold tracking-widest ${
                  line.type === "user" ? "text-zinc-500" : "text-[#d2ff00]"
                }`}>
                  {line.type === "user" ? "> USER_INPUT" : "AGASTYA_STREAM_OUTPUT"}
                </div>
                <p className={line.type === "user" ? "text-zinc-300" : "text-white"}>
                  {line.text}
                  {isStreaming && i === terminalLines.length - 1 && (
                    <span className="inline-block w-1.5 h-3.5 bg-[#d2ff00] ml-1 animate-pulse" />
                  )}
                </p>
              </div>
            ))}
          </div>

          {/* Query Input Bar */}
          <form onSubmit={handleExecute} className="mt-8 flex gap-3 border-t border-zinc-900 pt-4">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask a question or enter a code parameter query..."
              disabled={isStreaming}
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3 text-xs sm:text-sm text-zinc-200 focus:outline-none focus:border-zinc-700 transition-colors disabled:opacity-50 font-mono"
            />
            <button
              type="submit"
              disabled={isStreaming || !inputQuery.trim()}
              className="px-6 py-3 rounded-lg bg-[#d2ff00] font-mono font-bold text-xs uppercase tracking-wider text-black hover:bg-[#bcdc00] transition-colors disabled:opacity-30"
            >
              Execute
            </button>
          </form>
        </div>
      </section>

      {/* SECTION 3: SPECIFICATIONS GRID */}
      <section id="matrix-specs" className="max-w-[1400px] mx-auto px-6 py-24 border-t border-zinc-900 relative z-20">
        <div className="mb-14">
          <div className="font-mono text-[10px] tracking-widest text-zinc-500 uppercase mb-2">// SPECIFICATION TELEMETRY PROFILE</div>
          <h2 className="text-2xl font-bold tracking-tight uppercase text-white">Neural Network Matrix Specs</h2>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {specs.map((spec, index) => (
            <div 
              key={index} 
              className="p-6 rounded-xl border border-zinc-900 bg-zinc-950/40 backdrop-blur-sm hover:border-zinc-800 transition-colors group"
            >
              <div className="text-[9px] font-mono text-zinc-500 tracking-widest mb-3">{spec.label}</div>
              <div className="text-2xl font-extrabold tracking-tight text-white group-hover:text-[#d2ff00] transition-colors">
                {spec.value}
              </div>
              <div className="text-xs text-zinc-400 mt-1.5 font-sans">{spec.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* SECTION 4: DOCUMENTATION DEEP-DIVE FUNNEL */}
      <section className="max-w-[1400px] mx-auto px-6 pb-28 relative z-20">
        <div className="w-full rounded-2xl border border-zinc-900 bg-zinc-950 p-8 sm:p-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
          <div className="space-y-2.5">
            <div className="text-[10px] font-mono text-[#d2ff00] tracking-widest uppercase font-bold">
              // COMPLETE SYSTEMS ARCHITECTURE
            </div>
            <h3 className="text-xl font-bold tracking-tight uppercase text-white">
              Explore the Project Documentation
            </h3>
            <p className="text-zinc-400 text-sm max-w-2xl leading-relaxed font-sans">
              Review full descriptions of our Byte-Level BPE Tokenizer, the mathematical multi-head causal attention mask block definitions, and the MLOps token alignment rules that prevent system gibberish translation bugs.
            </p>
          </div>

          <Link 
            href="/docs" 
            className="w-full md:w-auto px-6 py-4 rounded-lg bg-white text-black font-mono font-bold text-xs tracking-wider uppercase text-center hover:bg-zinc-200 transition-colors whitespace-nowrap"
          >
            Open Documentation Manual ↗
          </Link>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-zinc-900 bg-black relative z-20">
        <div className="max-w-[1400px] mx-auto px-6 py-10 flex flex-col sm:flex-row items-center justify-between text-[10px] font-mono text-zinc-500 gap-4 uppercase tracking-widest text-center sm:text-left">
          <div>Copyright © Agastya Project System</div>
          <div className="flex flex-wrap justify-center gap-x-8 gap-y-2">
            <a href="#matrix-specs" className="hover:text-white transition-colors">Architecture</a>
            <Link href="/docs" className="hover:text-white transition-colors">Documentation</Link>
            <Link href="/article" className="hover:text-white transition-colors">Technical Article</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}