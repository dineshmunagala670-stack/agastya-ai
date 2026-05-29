"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";

interface TerminalLine {
  type: "user" | "ai" | "system";
  text: string;
}

export default function AgastyaNeumorphicWorkspace() {
  const [selectedEngine, setSelectedEngine] = useState<"v1" | "v2">("v2");
  const [inputQuery, setInputQuery] = useState("");
  const [apiBaseUrl, setApiBaseUrl] = useState("https://dinesh05976-agastya-ai.hf.space");
  const [isLocalHost, setIsLocalHost] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([
    { type: "ai", text: "I am Agastya, a neural network trained to process language naturally." },
    { type: "user", text: "Rewrite the given sentence using a different but similar word: She partook in the event." },
    { type: "ai", text: "The solution shored up on the following sentence: She chose to participate in the event." },
    { type: "user", text: "What is the weather like?" },
    { type: "ai", text: "I do not have access to live meteorological instruments, but it is an excellent day to optimize algorithms." }
  ]);

  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Dynamic system environment and browser tab configuration loop
  useEffect(() => {
    if (typeof window !== "undefined") {
      // 1. Force the Browser Tab Title text string exactly to "Agastya"
      document.title = "Agastya";

      // 2. Generate and inject the custom Vector Stream Logo directly into the Browser Tab Icon Slot
      const svgLogoContent = `
        <svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
          <g fill="none" stroke="#000000" stroke-width="28" stroke-linecap="round" stroke-linejoin="round">
            <path d="M 120 400 L 256 120"/>
            <path d="M 160 400 L 256 160"/>
            <path d="M 200 400 L 256 200"/>
            <path d="M 392 400 L 256 120"/>
            <path d="M 352 400 L 256 160"/>
            <path d="M 312 400 L 256 200"/>
            <path d="M 256 250 L 195 320" stroke-width="20"/>
            <path d="M 256 250 L 317 320" stroke-width="20"/>
            <circle cx="256" cy="115" r="14" fill="#000000" stroke="none"/>
          </g>
        </svg>
      `;
      
      const blob = new Blob([svgLogoContent], { type: "image/svg+xml" });
      const faviconUrl = URL.createObjectURL(blob);
      
      let linkElement: HTMLLinkElement | null = document.querySelector("link[rel*='icon']");
      if (!linkElement) {
        linkElement = document.createElement("link");
        linkElement.rel = "icon";
        document.head.appendChild(linkElement);
      }
      linkElement.href = faviconUrl;

      // 3. Parse active runtime domain context parameters
      const hostname = window.location.hostname;
      if (hostname === "localhost" || hostname === "127.0.0.1") {
        setApiBaseUrl("http://127.0.0.1:8000");
        setIsLocalHost(true);
        setTerminalLines((prev) => [
          ...prev,
          { type: "system", text: "System Status: Localhost gateway matrix connected on port 8000." }
        ]);
      } else {
        setApiBaseUrl("https://dinesh05976-agastya-ai.hf.space");
        setIsLocalHost(false);
      }
    }
  }, []);

  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [terminalLines]);

  const handleExecute = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isStreaming) return;

    const userPrompt = inputQuery;
    setTerminalLines((prev) => [...prev, { type: "user", text: userPrompt }]);
    setInputQuery("");
    setIsStreaming(true);

    if (selectedEngine === "v1") {
      // =====================================================================
      // 🟢 ENGINE MODE V1: AUTOREGRESSIVE CLOUD STREAM PIPELINE (38M MODEL)
      // =====================================================================
      setTerminalLines((prev) => [...prev, { type: "ai", text: "" }]);
      try {
        const response = await fetch("https://dinesh05976-agastya-ai.hf.space/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userPrompt }),
        });

        if (!response.ok) throw new Error();
        
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let compiledText = "";

        if (reader) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            compiledText += decoder.decode(value, { stream: true });
            setTerminalLines((prev) => {
              const copy = [...prev];
              copy[copy.length - 1] = { type: "ai", text: compiledText };
              return copy;
            });
          }
        }
      } catch (error) {
        setTerminalLines((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { type: "system", text: "Connection Failure: Unable to bind to live cloud V1 chat cluster stream." };
          return copy;
        });
      } finally {
        setIsStreaming(false);
      }
    } else {
      // =====================================================================
      // 🟡 ENGINE MODE V2: BIDIRECTIONAL SAFETENSORS MASK PREDICTION (55M MODEL)
      // =====================================================================
      if (!userPrompt.includes("[MASK]")) {
        setTerminalLines((prev) => [
          ...prev,
          { type: "system", text: "BERT Core Exception: The input prompt must contain a literal '[MASK]' parameter segment." }
        ]);
        setIsStreaming(false);
        return;
      }

      try {
        const response = await fetch(`${apiBaseUrl}/v2/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userPrompt }),
        });

        if (!response.ok) throw new Error();
        const data = await response.json();
        
        setTerminalLines((prev) => [
          ...prev,
          { type: "ai", text: `Prediction: ${data.prediction} | Core Matrices: [${data.tokens_extracted?.join(", ")}]` }
        ]);
      } catch (error) {
        setTerminalLines((prev) => [...prev, { type: "system", text: "Matrix Reflection Error: Safetensors node verification failed." }]);
      } finally {
        setIsStreaming(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#eef2f7] text-[#3a3d45] antialiased selection:bg-[#0071e3]/10 font-sans tracking-tight">
      
      {/* NEUMORPHIC FLOATING NAVIGATION HEADER */}
      <header className="w-full bg-[#eef2f7] border-b border-[#e2e8f0]/40 sticky top-0 z-50 px-6 py-3">
        <div className="max-w-[1300px] mx-auto px-5 h-12 rounded-xl bg-[#eef2f7] shadow-[4px_4px_10px_#cbd5e1,-4px_-4px_10px_#ffffff] flex items-center justify-between gap-4 text-xs">
          
          {/* LOGO AND BRANDING */}
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2 text-[13px] font-bold text-[#1d1d1f] hover:opacity-70 transition-opacity tracking-tight">
              {/* Native Apple-Style SVG Vector Stream Logo */}
              <svg className="w-4 h-4 text-black" viewBox="0 0 512 512" fill="none" stroke="currentColor" strokeWidth="28" strokeLinecap="round" strokeLinejoin="round">
                <path d="M 120 400 L 256 120"/>
                <path d="M 160 400 L 256 160"/>
                <path d="M 200 400 L 256 200"/>
                <path d="M 392 400 L 256 120"/>
                <path d="M 352 400 L 256 160"/>
                <path d="M 312 400 L 256 200"/>
                <path d="M 256 250 L 195 320" strokeWidth="20"/>
                <path d="M 256 250 L 317 320" strokeWidth="20"/>
                <circle cx="256" cy="115" r="14" fill="currentColor" stroke="none"/>
              </svg>
              <span>Agastya Labs</span>
            </Link>
            <nav className="hidden md:flex items-center gap-6 text-[#64748b] font-medium">
              <a href="#workspace" className="text-[#0071e3]">Workspace</a>
              <a href="#specs" className="hover:text-[#1d1d1f] transition-colors">Architecture</a>
              <Link href="/docs" className="hover:text-[#1d1d1f] transition-colors">Documentation</Link>
              <Link href="/article" className="hover:text-[#1d1d1f] transition-colors">Article</Link>
            </nav>
          </div>
          
          <div className="flex items-center gap-4 text-[11px]">
            <div className="flex items-center gap-1.5 px-3 py-1 bg-[#eef2f7] shadow-[inset_2px_2px_5px_#cbd5e1,inset_-2px_-2px_5px_#ffffff] rounded-full text-[#64748b] font-medium">
              <span className={`h-1.5 w-1.5 rounded-full ${isLocalHost ? "bg-amber-500" : "bg-[#34c759]"}`} />
              {isLocalHost ? "Local Testing Port" : "Online Cluster Operational"}
            </div>
          </div>
        </div>
      </header>

      {/* CORE WORKSPACE GRID CONTAINER */}
      <main id="workspace" className="max-w-[1300px] mx-auto px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* LEFT COLUMN: COMPACT COMPONENT SIDEBAR */}
        <div className="lg:col-span-4 space-y-5">
          
          {/* CONTROL BOX 1: MLOPS MANAGEMENT */}
          <div className="bg-[#eef2f7] shadow-[6px_6px_12px_#cbd5e1,-6px_-6px_12px_#ffffff] rounded-2xl p-4 space-y-3">
            <h2 className="text-[13px] font-bold text-[#1d1d1f] uppercase tracking-wide">
              MLOps Orchestration
            </h2>
            <p className="text-[#64748b] text-[11px] leading-relaxed">
              Hot-swap weight tracks dynamically during modular testing runs without experiencing websocket down-times.
            </p>
            <button className="w-full bg-[#eef2f7] shadow-[3px_3px_6px_#cbd5e1,-3px_-3px_6px_#ffffff] active:shadow-[inset_2px_2px_5px_#cbd5e1,inset_-2px_-2px_5px_#ffffff] text-[#0071e3] font-semibold text-[11px] py-2 rounded-xl transition-all uppercase tracking-wider">
              Sync Model Weights
            </button>
          </div>

          {/* CONTROL BOX 2: OPEN SOURCE */}
          <div className="bg-[#eef2f7] shadow-[6px_6px_12px_#cbd5e1,-6px_-6px_12px_#ffffff] rounded-2xl p-4 space-y-3">
            <h2 className="text-[13px] font-bold text-[#1d1d1f] uppercase tracking-wide">
              Open Source
            </h2>
            <p className="text-[#64748b] text-[11px] leading-relaxed">
              Examine underlying transformer multi-head weights layouts, vocabulary files, and local training code configurations.
            </p>
            <a 
              href="https://github.com/Dinesh05976/ai-builder" 
              target="_blank" 
              className="w-full block text-center bg-[#eef2f7] shadow-[3px_3px_6px_#cbd5e1,-3px_-3px_6px_#ffffff] active:shadow-[inset_2px_2px_5px_#cbd5e1,inset_-2px_-2px_5px_#ffffff] text-[#34c759] font-semibold text-[11px] py-2 rounded-xl transition-all uppercase tracking-wider"
            >
              View GitHub Repository
            </a>
          </div>

          {/* CONTROL BOX 3: COMPACT TELEMETRY */}
          <div className="bg-[#eef2f7] shadow-[6px_6px_12px_#cbd5e1,-6px_-6px_12px_#ffffff] rounded-2xl p-4 space-y-3">
            <h2 className="text-[13px] font-bold text-[#1d1d1f] uppercase tracking-wide">
              Model Telemetry Profile
            </h2>
            <div className="space-y-2 text-[11px]">
              <div className="flex justify-between items-center border-b border-[#e2e8f0]/60 pb-1.5">
                <span className="text-[#64748b]">Parameter Layout Count</span>
                <span className="text-[#1d1d1f] font-bold">{selectedEngine === "v2" ? "55,013,376" : "38,154,240"}</span>
              </div>
              <div className="flex justify-between items-center border-b border-[#e2e8f0]/60 pb-1.5">
                <span className="text-[#64748b]">Structural Attention Layers</span>
                <span className="text-[#1d1d1f] font-bold">{selectedEngine === "v2" ? "6 Blocks" : "12 Blocks"}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[#64748b]">Attention Heads Width</span>
                <span className="text-[#1d1d1f] font-bold">{selectedEngine === "v2" ? "12 Heads" : "6 Heads"}</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: RE-SIZED INTERACTIVE TERMINAL HUB */}
        <div className="lg:col-span-8 bg-[#eef2f7] shadow-[8px_8px_16px_#cbd5e1,-8px_-8px_16px_#ffffff] rounded-2xl p-5 flex flex-col justify-between min-h-[520px]">
          
          {/* PANEL CONTROLLER TOP BAR */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#cbd5e1]/40 pb-3 mb-4">
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-[#ff5f56]/80 shadow-sm" />
                <span className="w-2 h-2 rounded-full bg-[#ffbd2e]/80 shadow-sm" />
                <span className="w-2 h-2 rounded-full bg-[#27c93f]/80 shadow-sm" />
              </div>
              <span className="text-[#64748b] text-[11px] font-mono ml-1">agastya_inference_hub.sh</span>
            </div>
            
            {/* NEUMORPHIC ENGINE SELECTOR */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-[#64748b] font-medium uppercase tracking-wider">Deploy Target:</span>
              <select 
                value={selectedEngine}
                onChange={(e) => setSelectedEngine(e.target.value as "v1" | "v2")}
                className="bg-[#eef2f7] shadow-[2px_2px_5px_#cbd5e1,-2px_-2px_5px_#ffffff] border-none rounded-lg px-2.5 py-1 text-[11px] text-[#1d1d1f] font-semibold focus:outline-none cursor-pointer text-center"
              >
                <option value="v1">Agastya v1 (38M Causal Stream)</option>
                <option value="v2">Agastya v2 (55M BERT Safetensors)</option>
              </select>
            </div>
          </div>

          {/* INSET/SUNKEN VIEWPORT THREAD BOX */}
          <div 
            ref={scrollContainerRef}
            className="flex-1 bg-[#eef2f7] shadow-[inset_3px_3px_6px_#cbd5e1,inset_-3px_-3px_6px_#ffffff] rounded-xl p-4 space-y-4 overflow-y-auto max-h-[340px] text-[12px] leading-relaxed"
          >
            {terminalLines.map((line, i) => {
              if (line.type === "system") {
                return (
                  <div key={i} className="text-[#64748b] text-center text-[10px] font-medium bg-[#eef2f7] py-1 shadow-[inset_1px_1px_3px_#cbd5e1] rounded-md my-1 italic">
                    {line.text}
                  </div>
                );
              }
              return (
                <div 
                  key={i} 
                  className={`flex flex-col max-w-[85%] space-y-1 ${
                    line.type === "user" ? "ml-auto items-end" : "items-start"
                  }`}
                >
                  <div className="text-[9px] font-bold uppercase text-[#94a3b8] px-1 tracking-wider">
                    {line.type === "user" ? "Matrix Input" : "Agastya Core Engine"}
                  </div>
                  <div 
                    className={`rounded-xl px-3.5 py-2 border border-[#e2e8f0]/40 shadow-sm ${
                      line.type === "user" 
                        ? "bg-[#0071e3] text-white font-medium" 
                        : "bg-white text-[#3a3d45]"
                    }`}
                  >
                    {line.text}
                    {isStreaming && i === terminalLines.length - 1 && (
                      <span className="inline-block w-1.5 h-3.5 bg-[#0071e3] ml-1 animate-pulse" />
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* INSET DISPATCH CONTAINER PROMPT FORM */}
          <form onSubmit={handleExecute} className="mt-4 flex gap-3 border-t border-[#cbd5e1]/30 pt-3">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder={
                selectedEngine === "v1"
                  ? "Query streaming language parameters here..."
                  : "Pass context block with [MASK] (e.g., A pointer holds a memory [MASK] inside code.)..."
              }
              disabled={isStreaming}
              className="flex-1 bg-[#eef2f7] shadow-[inset_2px_2px_5px_#cbd5e1,inset_-2px_-2px_5px_#ffffff] rounded-xl px-3.5 py-2.5 text-[12px] text-[#1d1d1f] focus:outline-none disabled:opacity-50 placeholder-[#94a3b8]"
            />
            <button
              type="submit"
              disabled={isStreaming || !inputQuery.trim()}
              className="px-4 py-2.5 rounded-xl bg-[#eef2f7] shadow-[3px_3px_6px_#cbd5e1,-3px_-3px_6px_#ffffff] active:shadow-[inset_2px_2px_4px_#cbd5e1,inset_-2px_-2px_4px_#ffffff] text-[#0071e3] font-bold uppercase tracking-wider text-[11px] transition-all disabled:opacity-30"
            >
              Execute
            </button>
          </form>
        </div>
      </main>

      {/* TECHNICAL NETWORK PROFILE GRID DETAILS */}
      <section id="specs" className="max-w-[1300px] mx-auto px-6 py-10 border-t border-[#cbd5e1]/40">
        <div className="mb-6">
          <p className="text-[10px] font-bold text-[#94a3b8] uppercase tracking-widest mb-0.5">Specification Analytics Profiles</p>
          <h2 className="text-xl font-bold text-[#1d1d1f]">Neural Grid Projections</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {[
            { label: "Total Node Weight Layout", value: "55,013,376", desc: "V2 Bidirectional Core Architecture" },
            { label: "Context Sequence Frame", value: "256 Target Tokens", desc: "Dual Causal Attention Horizon Span" },
            { label: "Transformer Block Units", value: "6 Total Blocks", desc: "Pre-LayerNorm Structural Elements" },
            { label: "Execution Transformer Heads", value: "12 Projection Heads", desc: "Independent Attention Weight Windows" },
            { label: "Hidden Channel Vector Matrix", value: "768 Dimensions", desc: "Hidden Dimensional Vector Width Size" },
            { label: "Vocabulary Dictionary Density", value: "8,000 Key Entries", desc: "WordPiece Dictionary Configuration Model" }
          ].map((spec, index) => (
            <div key={index} className="p-4 rounded-xl bg-[#eef2f7] shadow-[4px_4px_8px_#cbd5e1,-4px_-4px_8px_#ffffff] border border-white/40">
              <div className="text-[10px] font-semibold text-[#64748b] mb-1">{spec.label}</div>
              <div className="text-xl font-bold text-[#1d1d1f]">
                {index === 0 ? (selectedEngine === "v2" ? "55,013,376" : "38,154,240") : spec.value}
              </div>
              <div className="text-[11px] text-[#94a3b8] mt-0.5">{spec.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-[#cbd5e1]/40 bg-[#eef2f7] py-6">
        <div className="max-w-[1300px] mx-auto px-6 flex flex-col sm:flex-row items-center justify-between text-[11px] text-[#94a3b8] gap-3">
          <div>Copyright © Agastya Cluster Systems. All structural rights reserved.</div>
          <div className="flex items-center gap-6 font-medium">
            <a href="#specs" className="hover:text-[#1d1d1f] transition-colors">Architecture</a>
            <Link href="/docs" className="hover:text-[#1d1d1f] transition-colors">Documentation</Link>
            <Link href="/article" className="hover:text-[#1d1d1f] transition-colors">Technical Article</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}