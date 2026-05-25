"use client";
import React, { useState, useEffect, useRef } from 'react';

export default function AgastyaShowcase() {
  const [messages, setMessages] = useState<{ sender: 'user' | 'ai', text: string }[]>([]);
  const [input, setInput] = useState('');
  const [apiStatus, setApiStatus] = useState({ 
    online: false, 
    device: 'OFFLINE', 
    horizon: '0',
    layers: '0',
    heads: '0',
    embedding: '0',
    vocabSize: '0',
    paramCount: '20,246,144'
  });
  const [syncStatus, setSyncStatus] = useState<'idle' | 'syncing' | 'success' | 'error'>('idle');
  const [loading, setLoading] = useState(false);

  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [messages]);

  const checkSystemHealth = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/health');
      if (res.ok) {
        const data = await res.json();
        setApiStatus({ 
          online: true, 
          device: data.device.toUpperCase(), 
          horizon: data.context_horizon.toString(),
          layers: data.n_layer.toString(),
          heads: data.n_head.toString(),
          embedding: data.n_embd.toString(),
          vocabSize: data.vocab_size.toString(),
          paramCount: data.param_count
        });
      } else {
        setApiStatus(prev => ({ ...prev, online: false, device: 'OFFLINE' }));
      }
    } catch {
      setApiStatus(prev => ({ ...prev, online: false, device: 'OFFLINE' }));
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

      if (!response.body) throw new Error("No response streaming pipeline available.");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      setMessages(prev => [...prev, { sender: 'ai', text: "" }]);
      setLoading(false);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const characterChunk = decoder.decode(value, { stream: true });
        
        setMessages(prev => {
          if (prev.length === 0) return prev;
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          
          updated[lastIndex] = {
            ...updated[lastIndex],
            text: updated[lastIndex].text + characterChunk
          };
          
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
    /* FIXED FIXED-HEIGHT VIEWPORT WRAPPER: Forced to fill screen dimension limits exactly without any root scaling leaks */
    <div className="h-screen w-screen bg-zinc-950 text-zinc-100 font-sans flex flex-col items-center p-6 overflow-hidden selection:bg-teal-500/30">
      
      {/* HEADER HUD STATUS SLOTS */}
      <header className="w-full max-w-5xl flex flex-col md:flex-row justify-between items-start md:items-center border-b border-zinc-800 pb-4 mb-4 gap-4 flex-shrink-0">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-teal-400 to-emerald-400 bg-clip-text text-transparent">
            PROJECT AGASTYA 20M
          </h1>
          <p className="text-zinc-400 text-xs mt-0.5">
            Custom Character-Level Autoregressive Transformer Platform
          </p>
        </div>

        <div className="flex flex-wrap gap-3 text-xs">
          <div className="bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-md">
            <span className="text-zinc-500 uppercase block font-semibold text-[10px] mb-0.5">Core Engine API</span>
            <span className={`font-bold flex items-center gap-1.5 ${apiStatus.online ? 'text-teal-400' : 'text-red-400'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${apiStatus.online ? 'bg-teal-400 animate-pulse' : 'bg-red-400'}`} />
              {apiStatus.online ? 'ONLINE (STREAMING)' : 'DISCONNECTED'}
            </span>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-md">
            <span className="text-zinc-500 uppercase block font-semibold text-[10px] mb-0.5">Compute Hardware</span>
            <span className="text-zinc-200 font-bold tracking-wide">{apiStatus.device}</span>
          </div>
        </div>
      </header>

      {/* WORKSPACE LAYOUT PANELS GRID */}
      <main className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-6 flex-grow min-h-0 overflow-hidden mb-2">
        
        {/* SIDEBAR OPERATING DECKS (Left Column) */}
        <section className="lg:col-span-1 flex flex-col gap-4 h-full min-h-0 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-zinc-900">
          
          {/* NEURAL NETWORK SPECIFICATION MATRIX CARD */}
          <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg flex-shrink-0">
            <h3 className="text-xs font-bold text-teal-400 tracking-wider uppercase border-b border-zinc-800 pb-1.5 mb-3">
              Neural Network Matrix Specs
            </h3>
            
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                <span className="text-[9px] text-zinc-500 block mb-0.5">TOTAL PARAMETERS</span>
                <span className="text-zinc-200 font-bold text-[11px]">{apiStatus.paramCount}</span>
              </div>
              <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                <span className="text-[9px] text-zinc-500 block mb-0.5">CONTEXT HORIZON</span>
                <span className="text-zinc-200 font-bold text-[11px]">{apiStatus.horizon} Tokens</span>
              </div>
              <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                <span className="text-[9px] text-zinc-500 block mb-0.5">ATTENTION LAYERS</span>
                <span className="text-zinc-200 font-bold text-[11px]">{apiStatus.layers} Blocks</span>
              </div>
              <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                <span className="text-[9px] text-zinc-500 block mb-0.5">ATTENTION HEADS</span>
                <span className="text-zinc-200 font-bold text-[11px]">{apiStatus.heads} Heads</span>
              </div>
              <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                <span className="text-[9px] text-zinc-500 block mb-0.5">HIDDEN CHANNELS</span>
                <span className="text-zinc-200 font-bold text-[11px]">{apiStatus.embedding} dims</span>
              </div>
              <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                <span className="text-[9px] text-zinc-500 block mb-0.5">VOCAB DICTIONARY</span>
                <span className="text-zinc-200 font-bold text-[11px]">{apiStatus.vocabSize} Tokens</span>
              </div>
            </div>
          </div>

          {/* MLOps BRAIN HOT SYNC CARD */}
          <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg flex flex-col justify-between flex-shrink-0">
            <div>
              <h3 className="text-xs font-bold text-zinc-300 tracking-wider uppercase border-b border-zinc-800 pb-1.5 mb-2">
                MLOps Weight Synchronization
              </h3>
              <p className="text-zinc-400 text-[11px] leading-relaxed mb-3">
                Hot-swap active GPU parameter layers on the fly once your fine-tuning pipeline reaches targets.
              </p>
            </div>
            
            <button
              onClick={handleSyncWeights}
              disabled={syncStatus === 'syncing' || !apiStatus.online}
              className={`w-full py-2 px-4 rounded text-xs font-medium tracking-wider transition-all duration-200 ${
                syncStatus === 'syncing' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40 cursor-wait' :
                syncStatus === 'success' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' :
                syncStatus === 'error' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                'bg-zinc-800 hover:bg-zinc-700 text-zinc-100 border border-zinc-700 active:scale-98 disabled:opacity-40'
              }`}
            >
              {syncStatus === 'idle' && 'SYNC NEWLY TRAINED BRAIN'}
              {syncStatus === 'syncing' && 'HOT-SWAPPING LAYERS...'}
              {syncStatus === 'success' && '✓ WEIGHT MATRIX ALIGNED'}
              {syncStatus === 'error' && '⚡ HOT SWAP REJECTED BY HOST'}
            </button>
          </div>

          {/* OPEN SOURCE COMMUNITY GATE CARD */}
          <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-lg flex flex-col justify-between flex-shrink-0">
            <div>
              <h3 className="text-xs font-bold text-zinc-300 tracking-wider uppercase border-b border-zinc-800 pb-1.5 mb-2">
                Open Source Architecture
              </h3>
              <p className="text-zinc-400 text-[11px] leading-relaxed mb-3">
                Review training logs, raw datasets compilation logic, or access code architecture blocks online.
              </p>
            </div>
            
            <a
              href="https://github.com/dineshmunagala670-stack/agastya-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="w-full py-2 px-4 rounded font-bold text-xs tracking-wider text-center transition-all bg-gradient-to-r from-teal-500 to-emerald-500 text-zinc-950 flex items-center justify-center gap-2 shadow-md hover:opacity-90"
            >
              WANT TO TRAIN THIS YOURSELF?
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
              </svg>
            </a>
          </div>
        </section>

        {/* TERMINAL INFECT CHAT PORT VIEW (Right Column - Dynamic Fluid Sizing) */}
        <section className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-lg flex flex-col h-full overflow-hidden shadow-2xl min-h-0">
          <div className="bg-zinc-950 border-b border-zinc-800 px-4 py-2.5 flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-2">
              <div className="flex gap-1.5">
                <span className="w-2 h-2 rounded-full bg-red-500/60" />
                <span className="w-2 h-2 rounded-full bg-amber-500/60" />
                <span className="w-2 h-2 rounded-full bg-emerald-500/60" />
              </div>
              <span className="text-zinc-500 text-xs font-mono ml-2">agastya_inference_loop.sh</span>
            </div>
            <span className="text-[10px] font-mono text-zinc-500 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
              TEMPERATURE=0.3
            </span>
          </div>

          {/* DYNAMIC SCROLL CONTAINER: Squeezes to fill remaining screen space cleanly */}
          <div 
            ref={chatContainerRef} 
            className="flex-grow overflow-y-auto p-4 flex flex-col gap-4 font-mono text-xs scrollbar-thin scrollbar-thumb-zinc-800 min-h-0"
          >
            {messages.length === 0 && (
              <div className="text-zinc-600 italic text-center my-auto p-8">
                // Streaming interface operational. Input parameters to execute real-time token projections.
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-md p-3 whitespace-pre-wrap break-words overflow-wrap-anywhere ${
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
          </div>

          {/* BOTTOM INTERACTION TRANSMISSION STRING FIELD */}
          <form onSubmit={handleSendMessage} className="p-3 bg-zinc-950 border-t border-zinc-800 flex gap-2 flex-shrink-0">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Query streaming language parameters here..."
              className="flex-grow bg-zinc-900 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-100 focus:outline-none focus:border-teal-500 transition-colors"
              disabled={loading || !apiStatus.online}
            />
            <button
              type="submit"
              disabled={loading || !apiStatus.online}
              className="bg-teal-600 hover:bg-teal-500 font-bold text-zinc-950 px-4 py-1.5 rounded-md text-xs tracking-wider transition-colors disabled:opacity-30 flex-shrink-0"
            >
              EXECUTE
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}