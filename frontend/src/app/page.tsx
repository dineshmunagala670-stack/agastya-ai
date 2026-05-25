"use client";
import React, { useState, useEffect, useRef } from 'react';

export default function AgastyaShowcase() {
  const [messages, setMessages] = useState<{ sender: 'user' | 'ai', text: string }[]>([]);
  const [input, setInput] = useState('');
  const [apiStatus, setApiStatus] = useState({ online: false, device: 'unknown', horizon: '0' });
  const [syncStatus, setSyncStatus] = useState<'idle' | 'syncing' | 'success' | 'error'>('idle');
  const [loading, setLoading] = useState(false);

  // Auto-scroll Anchor Ref
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  // Trigger smooth scroll down every single time the text stream mutates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const checkSystemHealth = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/health');
      if (res.ok) {
        const data = await res.json();
        setApiStatus({ online: true, device: data.device.toUpperCase(), horizon: data.context_horizon });
      } else {
        setApiStatus({ online: false, device: 'OFFLINE', horizon: '0' });
      }
    } catch {
      setApiStatus({ online: false, device: 'OFFLINE', horizon: '0' });
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input;
    setInput('');
    setMessages(prev => [...prev, { sender: 'user', text: userText }]);
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText }),
      });

      if (!response.body) throw new Error("No readable response pipeline stream active.");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      setMessages(prev => [...prev, { sender: 'ai', text: "" }]);
      setLoading(false);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const characterChunk = decoder.decode(value, { stream: true });
        
        setMessages(prev => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[updated.length - 1].text += characterChunk;
          }
          return updated;
        });
      }
    } catch {
      setMessages(prev => [...prev, { sender: 'ai', text: "[Failed to read streaming token chunks from core neural network]" }]);
      setLoading(false);
    }
  };

  const handleSyncWeights = async () => {
    setSyncStatus('syncing');
    try {
      const response = await fetch('http://127.0.0.1:8000/reload', { method: 'POST' });
      if (response.ok) {
        setSyncStatus('success');
        setTimeout(() => setSyncStatus('idle'), 3000);
        await checkSystemHealth();
      } else {
        setSyncStatus('error');
      }
    } catch {
      setSyncStatus('error');
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans flex flex-col items-center p-6 selection:bg-teal-500/30">
      
      {/* HEADER HUD CONTAINER */}
      <header className="w-full max-w-5xl flex flex-col md:flex-row justify-between items-start md:items-center border-b border-zinc-800 pb-6 mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-teal-400 to-emerald-400 bg-clip-text text-transparent">
            PROJECT AGASTYA 20M
          </h1>
          <p className="text-zinc-400 text-sm mt-1">
            Custom Character-Level Autoregressive Transformer Platform
          </p>
        </div>

        <div className="flex flex-wrap gap-3 text-xs">
          <div className="bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-md">
            <span className="text-zinc-500 uppercase block font-semibold mb-0.5">Core Engine API</span>
            <span className={`font-bold flex items-center gap-1.5 ${apiStatus.online ? 'text-teal-400' : 'text-red-400'}`}>
              <span className={`w-2 h-2 rounded-full ${apiStatus.online ? 'bg-teal-400 animate-pulse' : 'bg-red-400'}`} />
              {apiStatus.online ? 'ONLINE (STREAMING)' : 'DISCONNECTED'}
            </span>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-md">
            <span className="text-zinc-500 uppercase block font-semibold mb-0.5">Compute Hardware</span>
            <span className="text-zinc-200 font-bold tracking-wide">{apiStatus.device}</span>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-md">
            <span className="text-zinc-500 uppercase block font-semibold mb-0.5">Memory Block Horizon</span>
            <span className="text-zinc-200 font-bold">{apiStatus.horizon} Tokens</span>
          </div>
        </div>
      </header>

      {/* MAIN WORKSPACE INTERFACE */}
      <main className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-8 flex-grow">
        
        {/* CONTROL DECK CONTROLLERS (LEFT SIDEBAR) */}
        <section className="lg:col-span-1 flex flex-col gap-6">
          
          {/* MLOps SYNC MODULE CARD */}
          <div className="bg-zinc-900 border border-zinc-800 p-5 rounded-lg flex flex-col justify-between h-fit">
            <div>
              <h3 className="text-sm font-bold text-zinc-300 tracking-wide uppercase border-b border-zinc-800 pb-2 mb-4">
                Dynamic MLOps Orchestration
              </h3>
              <p className="text-zinc-400 text-xs leading-relaxed mb-4">
                When you run iterative fine-tuning sequences to feed Agastya fresh story sets, use the matrix control below to swap parameters on the fly without breaking active web sockets.
              </p>
            </div>
            
            <button
              onClick={handleSyncWeights}
              disabled={syncStatus === 'syncing' || !apiStatus.online}
              className={`w-full py-2.5 px-4 rounded-md font-medium text-xs tracking-wider transition-all duration-200 flex items-center justify-center gap-2 ${
                syncStatus === 'syncing' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40 cursor-wait' :
                syncStatus === 'success' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' :
                syncStatus === 'error' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                'bg-zinc-800 hover:bg-zinc-700 text-zinc-100 border border-zinc-700 active:scale-98 disabled:opacity-40 disabled:pointer-events-none'
              }`}
            >
              {syncStatus === 'idle' && 'SYNC NEWLY TRAINED BRAIN'}
              {syncStatus === 'syncing' && 'HOT-SWAPPING LAYERS...'}
              {syncStatus === 'success' && '✓ WEIGHT MATRIX ALIGNED'}
              {syncStatus === 'error' && '⚡ HOT SWAP REJECTED BY HOST'}
            </button>
          </div>

          {/* NEW: OPEN SOURCE COMMUNITY CALLOUT CARD */}
          <div className="bg-zinc-900 border border-zinc-800 p-5 rounded-lg flex flex-col justify-between h-fit relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-24 h-24 bg-teal-500/5 rounded-full blur-xl group-hover:bg-teal-500/10 transition-all" />
            <div>
              <h3 className="text-sm font-bold text-teal-400 tracking-wide uppercase border-b border-zinc-800 pb-2 mb-4">
                Open Source Architecture
              </h3>
              <p className="text-zinc-400 text-xs leading-relaxed mb-5">
                Want to see the underlying 12-layer Transformer layout, character vocabulary token mappings, and training code configurations?
              </p>
            </div>
            
            <a
              href="https://github.com/dineshmunagala670-stack/agastya-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="w-full py-2.5 px-4 rounded-md font-bold text-xs tracking-wider text-center transition-all duration-200 bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-zinc-950 active:scale-98 shadow-lg shadow-teal-950/20 flex items-center justify-center gap-2"
            >
              WANT TO TRAIN THIS YOURSELF?
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
              </svg>
            </a>
          </div>

          {/* MODEL TELEMETRY METRICS HUD */}
          <div className="bg-zinc-900/50 border border-zinc-900 p-5 rounded-lg text-xs text-zinc-500 space-y-2">
            <span className="font-bold text-zinc-400 uppercase tracking-wider block mb-1">Model Telemetry Metrics</span>
            <div className="flex justify-between border-b border-zinc-900 pb-1.5"><span>Parameter Layout Count</span><span className="font-mono text-zinc-400">20,246,144</span></div>
            <div className="flex justify-between border-b border-zinc-900 pb-1.5"><span>Structural Attention Layers</span><span className="font-mono text-zinc-400">12 Blocks</span></div>
            <div className="flex justify-between"><span>Attention Heads Width</span><span className="font-mono text-zinc-400">6 Heads</span></div>
          </div>
        </section>

        {/* WORKSPACE CHAT TERMINAL CONTAINER */}
        <section className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-lg flex flex-col min-h-[500px] h-[600px] overflow-hidden shadow-2xl">
          <div className="bg-zinc-950 border-b border-zinc-800 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500/60" />
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/60" />
              </div>
              <span className="text-zinc-500 text-xs font-mono ml-2">agastya_inference_loop.sh</span>
            </div>
            <span className="text-[10px] font-mono text-zinc-500 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
              TEMPERATURE=0.3
            </span>
          </div>

          <div className="flex-grow overflow-y-auto p-4 flex flex-col gap-4 font-mono text-xs scrollbar-thin scrollbar-thumb-zinc-800">
            {messages.length === 0 && (
              <div className="text-zinc-600 italic text-center my-auto p-8">
                // Streaming interface ready. Input prompts to observe real-time token projection.
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-md p-3 whitespace-pre-wrap ${
                  msg.sender === 'user' 
                    ? 'bg-zinc-800 text-teal-400 border border-zinc-700 font-semibold' 
                    : 'bg-zinc-950 text-zinc-300 border border-zinc-850 shadow-inner'
                }`}>
                  <span className="text-[10px] uppercase font-bold block opacity-40 mb-1">
                    {msg.sender === 'user' ? '► USER_INPUT' : '▲ AGASTYA_STREAM_OUTPUT'}
                  </span>
                  {msg.text}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-zinc-950 text-zinc-500 border border-zinc-850 rounded-md p-3 font-mono">
                  ▲ Initializing character matrix stream...
                </div>
              </div>
            )}
            
            {/* Hidden Auto-Scroll Target Element */}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSendMessage} className="p-4 bg-zinc-950 border-t border-zinc-800 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Query streaming language parameters here..."
              className="flex-grow bg-zinc-900 border border-zinc-800 rounded-md px-3 py-2 text-xs font-mono text-zinc-100 focus:outline-none focus:border-teal-500 transition-colors"
              disabled={loading || !apiStatus.online}
            />
            <button
              type="submit"
              disabled={loading || !apiStatus.online}
              className="bg-teal-600 hover:bg-teal-500 font-bold text-zinc-950 px-4 py-2 rounded-md text-xs tracking-wider transition-colors disabled:opacity-30 disabled:pointer-events-none active:scale-97"
            >
              EXECUTE
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}