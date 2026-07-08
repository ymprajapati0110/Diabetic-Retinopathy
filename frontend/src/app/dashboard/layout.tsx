"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Eye, LogOut, Loader2 } from 'lucide-react';
import Link from 'next/link';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [profile, setProfile] = useState<any>(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      router.push('/login');
    } else {
      // Decode user name from token or set default
      setProfile({ name: "Dr. Clinician", role: "verified" });
    }
  }, [router]);

  const handleSignOut = () => {
    localStorage.removeItem('token');
    router.push('/login');
  };

  if (!profile) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-400 font-medium">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin mb-4" />
        <p className="text-sm tracking-wide">Initializing secure session...</p>
      </div>
    );
  }

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
          <div className="hidden sm:flex flex-col text-right">
            <span className="text-xs font-bold text-slate-200">{profile.name}</span>
            <span className="text-[9px] font-bold text-indigo-400 uppercase tracking-wider">Authorized Practitioner</span>
          </div>
          
          <div className="w-px h-6 bg-slate-900 hidden sm:block"></div>

          <button
            onClick={handleSignOut}
            className="group text-xs font-bold text-slate-400 hover:text-white bg-slate-900 hover:bg-slate-800 px-3 py-2 rounded-xl border border-slate-800 flex items-center gap-2 transition-all cursor-pointer"
          >
            <LogOut size={14} className="text-slate-500 group-hover:text-red-400 transition-colors" />
            Sign Out
          </button>
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
