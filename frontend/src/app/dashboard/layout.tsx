"use client";

import { useState, useEffect } from 'react';
import { Eye, Activity } from 'lucide-react';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [profile] = useState<any>({ name: "Dr. Clinician", role: "verified" });

  useEffect(() => {
    if (typeof window !== 'undefined' && !localStorage.getItem('token')) {
      localStorage.setItem('token', 'session-active');
    }
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased selection:bg-indigo-500/30">
      {/* Decorative Blur Background Circles */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-indigo-600/5 rounded-full blur-[120px] pointer-events-none -translate-y-1/2"></div>
      <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-emerald-600/5 rounded-full blur-[150px] pointer-events-none translate-y-1/3"></div>

      {/* Simplified Top Navigation Header */}
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-md border-b border-slate-900 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Eye className="text-white" size={22} />
          </div>
          <div>
            <h1 className="text-base font-black tracking-tight text-white">
              Retinal Diagnostic Portal
            </h1>
            <p className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">Clinical Screening Suite</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Neural Engine Active</span>
          </div>

          <div className="w-px h-6 bg-slate-900 hidden sm:block"></div>

          <div className="hidden sm:flex flex-col text-right">
            <span className="text-xs font-bold text-slate-200">{profile.name}</span>
            <span className="text-[9px] font-bold text-indigo-400 uppercase tracking-wider">Authorized Practitioner</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-6xl mx-auto px-6 py-8 relative z-10">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-[9px] text-slate-600 font-bold uppercase tracking-widest">
        © 2026 Retinal Screening Portal • Confidential Diagnostic Tool
      </footer>
    </div>
  );
}
