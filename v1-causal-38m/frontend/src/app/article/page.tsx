import type { Metadata } from "next";
import Link from "next/link";

// 🚀 1. SEARCH ENGINE OPTIMIZATION METADATA COLLATERAL
export const metadata: Metadata = {
  title: "Building a 38M Parameter Causal Transformer From Scratch | Agastya AI",
  description: "An in-depth structural analysis detailing how to design, train, and deploy a custom 38-million parameter language model using pure PyTorch and a free cloud architecture.",
  keywords: [
    "Build transformer from scratch",
    "PyTorch language model tutorial",
    "38M parameter LLM architecture",
    "Custom BPE tokenizer design",
    "Free AI model deployment hosting",
    "FastAPI streaming text tokens",
    "MLOps synchronization rules",
    "Agastya Labs Dinesh"
  ],
  authors: [{ name: "Dinesh" }],
  creator: "Dinesh",
  openGraph: {
    title: "How We Built Project Agastya: A 38M Parameter Transformer",
    description: "The complete technical breakdown of an autoregressive deep learning model engineered directly out of raw tensor matrices.",
    url: "https://agastya-ai.vercel.app/article",
    siteName: "Agastya Labs",
    type: "article",
    publishedTime: "2026-05-25T00:00:00.000Z",
    authors: ["Dinesh"],
  },
  twitter: {
    card: "summary_large_image",
    title: "Deep Dive: Coding a Causal LLM Layer-by-Layer",
    description: "Tracking the complete creation journey from initializing randomized weight blocks to streaming tokens live on free CPU tiers.",
  },
};

export default function AgastyaSEOArticlePage() {
  
  // 🧠 2. GOOGLE STRUCTURED DATA SCHEMA (JSON-LD)
  // This tells Google search bots exactly how to parse your page as an official TechArticle
  const jsonLdSchema = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    "headline": "Building a 38M Parameter Causal Transformer From Scratch",
    "description": "An in-depth structural analysis detailing how to design, train, and deploy a custom 38-million parameter language model using pure PyTorch and a free cloud architecture.",
    "inLanguage": "en-US",
    "mainEntityOfPage": "https://agastya-ai.vercel.app/article",
    "datePublished": "2026-05-25T00:00:00.000Z",
    "dateModified": "2026-05-25T00:00:00.000Z",
    "author": {
      "@type": "Person",
      "name": "Dinesh"
    },
    "publisher": {
      "@type": "Organization",
      "name": "Agastya Labs",
      "logo": {
        "@type": "ImageObject",
        "url": "https://agastya-ai.vercel.app/logo.png"
      }
    },
    "proficiencyLevel": "Advanced"
  };

  return (
    <div className="min-h-screen bg-black text-zinc-300 font-sans antialiased selection:bg-[#d2ff00]/30 selection:text-black">
      
      {/* INJECT SCHEMA MARKUP IN THE HEAD/BODY FOR GOOGLE RANKINGS */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLdSchema) }}
      />

      {/* MINIMAL BRAND HEADER WITH FUNNEL ACCENTS */}
      <header className="w-full border-b border-zinc-900 bg-black/50 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <Link href="/" className="font-mono text-xs tracking-widest text-white uppercase hover:text-zinc-400 transition-colors">
          // AGASTYA LABS
        </Link>
        <Link href="/" className="bg-[#d2ff00] text-black font-mono font-bold uppercase text-[10px] tracking-[0.15em] rounded-full px-5 py-2 hover:bg-[#bcdc00] transition-colors shadow-lg shadow-[#d2ff00]/15">
          Launch Live Workspace →
        </Link>
      </header>

      {/* ARTICLE CONTENT CANVAS */}
      <main className="max-w-3xl mx-auto px-6 py-16 sm:py-24 space-y-12">
        
        {/* ARTICLE HEADER BLOCK */}
        <header className="space-y-4 border-b border-zinc-900 pb-8">
          <div className="inline-block font-mono text-[10px] tracking-[0.25em] text-[#d2ff00] uppercase font-bold bg-zinc-950 px-3 py-1 border border-zinc-900 rounded">
            Engineering Documentation / Deep Learning
          </div>
          <h1 className="text-3xl sm:text-5xl font-black text-white uppercase tracking-tight leading-none pt-2">
            Building a 38M Parameter Causal Transformer From Scratch
          </h1>
          <div className="flex items-center gap-6 text-[11px] font-mono text-zinc-500 pt-2">
            <span>BY DINESH</span>
            <span>•</span>
            <span>MAY 25, 2026</span>
            <span>•</span>
            <span>12 MIN READ</span>
          </div>
        </header>

        {/* COMPACT INTERACTIVE FUNNEL CARD (ANCHOR VALUE) */}
        <aside className="w-full rounded-xl border border-zinc-800 bg-zinc-950/50 p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-dashed">
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-white uppercase tracking-wide">Want to test the active parameter weights?</h3>
            <p className="text-zinc-400 text-xs font-sans">We deployed this exact model setup to a free production stream backend.</p>
          </div>
          <Link href="/" className="text-xs font-mono font-bold text-[#d2ff00] hover:underline whitespace-nowrap">
            [ Run Live Inference Loop ]
          </Link>
        </aside>

        {/* CORE SEMANTIC TEXT MODULES */}
        <article className="space-y-8 text-sm sm:text-base leading-relaxed text-zinc-400 font-sans">
          
          <section className="space-y-4">
            <h2 className="text-xl sm:text-2xl font-bold text-white uppercase tracking-tight font-mono">
              1. The Inseparable Blueprint: Tokenizer and Weight Symbiosis
            </h2>
            <p>
              In modern machine learning production frameworks, architectural failures rarely originate from clean tensor multiplication blocks. Instead, the most common pitfall is structural data misalignment—specifically, a mismatch between the <strong>Tokenizer layout (.json)</strong> and the <strong>Weight matrices binary (.pth)</strong>.
            </p>
            <p>
              An autoregressive language model behaves entirely like a mathematical mapping operation. It maps localized strings into an intermediate integer token sequence, optimizes those sequences using specialized multi-head linear spaces, and decodes them back to readable language markers. If you compile a fresh sub-word dictionary utilizing Byte-Level Byte Pair Encoding (BPE) without immediately re-aligning your matrix nodes through gradient descent, the index tokens change, causing the neural engine to stream corrupted character combinations—commonly identified as <em>gibberish output arrays</em>.
            </p>
          </section>

          <section className="space-y-4">
            <h2 className="text-xl sm:text-2xl font-bold text-white uppercase tracking-tight font-mono">
              2. Structural Network Layout Dimensions
            </h2>
            <p>
              Project Agastya was designed from the ground up to minimize active server memory signatures while maintaining coherent computer science vocabulary mappings. Its architectural dimensions are frozen at these specifications:
            </p>
            <ul className="space-y-2 font-mono text-xs pl-4 border-l border-zinc-800 text-zinc-300">
              <li><strong className="text-white">Total Parameters:</strong> 38,154,240 (~38M structural weights)</li>
              <li><strong className="text-white">Causal Blocks Depth:</strong> 12 Sequential Pre-LN Layers</li>
              <li><strong className="text-white">Subspace Attention Heads:</strong> 8 Parallel Heads</li>
              <li><strong className="text-white">Hidden Embedding Width:</strong> 512 Vector Channels</li>
              <li><strong className="text-white">Context Horizon Span:</strong> 256 Active Token Windows</li>
              <li><strong className="text-white">Vocabulary Density:</strong> 2,000 Trained BPE Allocations</li>
            </ul>
          </section>

          <section className="space-y-4">
            <h2 className="text-xl sm:text-2xl font-bold text-white uppercase tracking-tight font-mono">
              3. Decoupled Free Cloud Container Execution
            </h2>
            <p>
              The complete distribution framework is deployed using a cost-free model pipeline split across independent architectures. Because the unified weight payload requires a tiny **~184.80 MB VRAM/RAM** footprint, expensive graphical processing nodes are completely unnecessary during public runtime operations.
            </p>
            <p>
              The backend architecture is packaged inside a custom Docker layout and deployed onto the free **Hugging Face Spaces CPU Basic Tier**. Running a specialized FastAPI web server container, the backend utilizes network handshakes to dynamically stream token sequences token-by-token directly to a serverless front-end application hosted on **Vercel**. This next-generation structure achieves true asynchronous, zero-cost production scaling with maximum viewport scannability.
            </p>
          </section>

        </article>

        {/* FINAL REDIRECT FUNNEL CALL TO ACTION BOX */}
        <section className="w-full border-t border-zinc-900 pt-12 text-center space-y-6">
          <div className="space-y-2">
            <h2 className="text-2xl font-extrabold text-white uppercase tracking-tight">Experience Agastya Live</h2>
            <p className="text-zinc-400 text-xs sm:text-sm max-w-xl mx-auto font-sans leading-relaxed">
              Test out the responsive brutalist code editor layout, view active generation metrics, and interact directly with the 38M parameter transformer engine right now.
            </p>
          </div>
          <div>
            <Link 
              href="/" 
              className="inline-block bg-[#d2ff00] text-black text-xs font-mono font-bold tracking-[0.15em] uppercase rounded-full px-8 py-4 hover:bg-[#bcdc00] transition-colors shadow-xl shadow-[#d2ff00]/10 transform active:scale-95 duration-150"
            >
              ← Open Interactive Chat Terminal
            </Link>
          </div>
        </section>

      </main>

      {/* MINIMAL METADATA FOOTER */}
      <footer className="w-full border-t border-zinc-950 bg-black py-8 text-center text-[10px] font-mono text-zinc-600 tracking-widest uppercase">
        © 2026 Agastya Project System / Engineered by Dinesh
      </footer>
    </div>
  );
}