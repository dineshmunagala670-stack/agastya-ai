// frontend/src/app/page.tsx
'use client';

import React, { useState, useRef, useEffect } from 'react';

// Explicitly define the message object blueprint structure for TypeScript
interface Message {
  id: string;
  text: string;
  isUser: boolean;
  promptOrigin?: string;
}

export default function Home() {
  // Bind the state to our custom Message array type
  const [messages, setMessages] = useState<Message[]>([
    { 
      id: 'init-node', 
      text: 'Interactive Training Hub Online. Test my outputs, flag structural errors, and submit corrections to update my parameter weights.', 
      isUser: false, 
      promptOrigin: '' 
    }
  ]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRetraining, setIsRetraining] = useState<boolean>(false);
  const [correctionTarget, setCorrectionTarget] = useState<string | null>(null); 
  const [correctionText, setCorrectionText] = useState<string>('');
  
  // Provide explicit typing for the HTML Div container ref element
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Type the form submission event correctly using React's event types
  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const userText = input.trim();
    if (!userText || isLoading || isRetraining) return;

    setMessages(prev => [...prev, { id: `u-${Date.now()}`, text: userText, isUser: true }]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { id: `a-${Date.now()}`, text: data.response, isUser: false, promptOrigin: userText }]);
    } catch {
      setMessages(prev => [...prev, { id: `e-${Date.now()}`, text: 'System offline: Verify that your Python API server is actively listening.', isUser: false }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCorrectionSubmit = async (msgId: string, originalPrompt: string) => {
    const trimmedCorrection = correctionText.trim();
    if (!trimmedCorrection) return;
    try {
      const res = await fetch('http://127.0.0.1:8000/submit-correction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: originalPrompt, correction: trimmedCorrection })
      });
      const data = await res.json();
      if (data.status === 'success') {
        alert('Data vector submitted successfully to dataset/input.txt!');
        setCorrectionTarget(null);
        setCorrectionText('');
      }
    } catch {
      alert('Error transmitting correction payload.');
    }
  };

  const handleTriggerRetrain = async () => {
    setIsRetraining(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/retrain', { method: 'POST' });
      const data = await res.json();
      alert(data.message + ' Giving your RTX 4050 about 15 seconds to run optimization sweeps.');
    } catch {
      alert('Error communicating with background optimizer thread.');
      setIsRetraining(false);
    }
  };

  useEffect(() => {
    if (isRetraining) {
      const timer = setTimeout(() => {
        setIsRetraining(false);
      }, 15000);
      return () => clearTimeout(timer);
    }
  }, [isRetraining]);

  return (
    <div className="bg-zinc-950 text-zinc-100 min-h-screen flex flex-col font-sans">
      {/* Control System Header Dashboard */}
      <header className="border-b border-zinc-900 bg-zinc-900/20 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`h-2.5 w-2.5 rounded-full ${isRetraining ? 'bg-amber-500 animate-spin' : 'bg-emerald-500 animate-pulse'}`}></div>
          <h1 className="text-sm font-bold tracking-widest text-white">AGASTYA TRAINING OPERATIONS</h1>
        </div>
        <button 
          onClick={handleTriggerRetrain}
          disabled={isRetraining || isLoading}
          className="bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-800 text-zinc-950 disabled:text-zinc-600 px-4 py-1.5 rounded-lg text-xs font-bold tracking-wider uppercase transition-all cursor-pointer disabled:cursor-not-allowed"
        >
          {isRetraining ? 'Re-Mapping Weights...' : 'Hot-Reload New Feedback Data'}
        </button>
      </header>

      {/* Primary Communication Canvas */}
      <main className="flex-1 max-w-4xl w-full mx-auto p-4 flex flex-col overflow-hidden h-[calc(100vh-69px)]">
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto space-y-6 pr-2 pb-6 scrollbar-thin">
          {messages.map((msg) => (
            <div key={msg.id} className="space-y-2">
              <div className={`flex gap-3 max-w-[85%] ${msg.isUser ? 'ml-auto flex-row-reverse' : ''}`}>
                <div className={`h-8 w-8 rounded-full flex items-center justify-center font-bold text-xs text-white shrink-0 ${msg.isUser ? 'bg-zinc-800' : 'bg-indigo-600'}`}>
                  {msg.isUser ? 'ME' : 'AG'}
                </div>
                <div className={`${msg.isUser ? 'bg-indigo-600 text-white' : 'bg-zinc-900 border border-zinc-800'} rounded-2xl px-4 py-2.5 text-sm shadow-md whitespace-pre-wrap`}>
                  {msg.text}
                </div>
              </div>

              {/* Interactive Training Panel Block Component */}
              {!msg.isUser && msg.promptOrigin && (
                <div className="pl-11">
                  {correctionTarget !== msg.id ? (
                    <button 
                      onClick={() => setCorrectionTarget(msg.id)}
                      className="text-[11px] text-indigo-400 hover:text-indigo-300 underline cursor-pointer"
                    >
                      [ Correct Response ]
                    </button>
                  ) : (
                    <div className="bg-zinc-900 border border-dashed border-zinc-800 p-3 rounded-xl mt-2 max-w-[80%] space-y-2">
                      <span className="text-[10px] text-amber-500 font-bold uppercase tracking-wider block">Alignment Input Editor</span>
                      <textarea
                        value={correctionText}
                        onChange={(e) => setCorrectionText(e.target.value)}
                        placeholder="Type the exact target string Agastya should produce..."
                        className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-100 focus:outline-none focus:border-amber-500 placeholder-zinc-600"
                        rows={2}
                      />
                      <div className="flex gap-2 justify-end">
                        <button type="button" onClick={() => setCorrectionTarget(null)} className="px-2 py-1 bg-zinc-800 text-[10px] rounded hover:bg-zinc-750 cursor-pointer">Cancel</button>
                        <button type="button" onClick={() => handleCorrectionSubmit(msg.id, msg.promptOrigin || '')} className="px-2 py-1 bg-amber-600 text-zinc-950 font-bold text-[10px] rounded hover:bg-amber-500 cursor-pointer">Submit Vector</button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3 text-zinc-500 text-sm italic pl-11 animate-pulse">
              Agastya is computing active attention routes...
            </div>
          )}
        </div>

        {/* Input Interactive Segment */}
        <footer className="mt-auto pt-4 border-t border-zinc-900 bg-zinc-950">
          <form onSubmit={handleChatSubmit} className="flex gap-2">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isRetraining ? "Weights are locked during active background fine-tuning..." : "Test dialogue responses..."}
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 placeholder-zinc-600"
              disabled={isLoading || isRetraining}
            />
            <button 
              type="submit" 
              disabled={isLoading || isRetraining || !input.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-900 disabled:text-zinc-700 font-medium text-sm rounded-xl px-6 transition-colors cursor-pointer"
            >
              Send
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}