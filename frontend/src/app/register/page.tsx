"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ShieldCheck, User, Lock, Mail, Eye } from 'lucide-react';
import api from '@/lib/api';

export default function RegisterPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    medical_license: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await api.post('/auth/register', formData);
      router.push('/login?registered=true');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4 relative overflow-hidden antialiased selection:bg-indigo-500/30">
      {/* Dynamic Glow Circles */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-indigo-600/10 rounded-full blur-[120px] pointer-events-none -translate-y-1/2"></div>
      <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-emerald-600/5 rounded-full blur-[150px] pointer-events-none translate-y-1/3"></div>

      <div className="w-full max-w-md bg-slate-900/40 backdrop-blur-xl border border-slate-900 p-8 md:p-10 rounded-3xl shadow-2xl relative z-10">
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 bg-gradient-to-tr from-indigo-600 to-indigo-400 rounded-xl flex items-center justify-center text-white shadow-lg shadow-indigo-500/20 mb-4 animate-pulse">
            <Eye size={24} />
          </div>
          <h1 className="text-2xl font-black tracking-tight text-white">Create Account</h1>
          <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mt-1">Retinal Screening Portal</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-500/10 text-red-400 rounded-2xl text-xs border border-red-500/20 flex items-start gap-3">
            <ShieldCheck className="shrink-0 mt-0.5 text-red-400" size={18} />
            <p className="font-medium leading-relaxed">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest ml-1">Full Name</label>
            <div className="relative">
              <User className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
              <input 
                type="text" 
                required
                className="w-full pl-11 pr-4 py-3 bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl focus:ring-2 focus:ring-indigo-500/20 outline-none transition-all placeholder:text-slate-600 text-sm"
                placeholder="Dr. Jane Doe"
                value={formData.name}
                onChange={e => setFormData({...formData, name: e.target.value})}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest ml-1">Work Email</label>
            <div className="relative">
              <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
              <input 
                type="email" 
                required
                className="w-full pl-11 pr-4 py-3 bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl focus:ring-2 focus:ring-indigo-500/20 outline-none transition-all placeholder:text-slate-600 text-sm"
                placeholder="jane.doe@hospital.org"
                value={formData.email}
                onChange={e => setFormData({...formData, email: e.target.value})}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest ml-1">Medical License #</label>
            <div className="relative">
              <ShieldCheck className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
              <input 
                type="text" 
                required
                className="w-full pl-11 pr-4 py-3 bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl focus:ring-2 focus:ring-indigo-500/20 outline-none transition-all placeholder:text-slate-600 text-sm"
                placeholder="MD-12345678"
                value={formData.medical_license}
                onChange={e => setFormData({...formData, medical_license: e.target.value})}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest ml-1">Password</label>
            <div className="relative">
              <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
              <input 
                type="password" 
                required
                className="w-full pl-11 pr-4 py-3 bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl focus:ring-2 focus:ring-indigo-500/20 outline-none transition-all placeholder:text-slate-600 text-sm tracking-widest"
                placeholder="••••••••"
                value={formData.password}
                onChange={e => setFormData({...formData, password: e.target.value})}
              />
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-bold transition-all shadow-lg shadow-indigo-500/10 hover:shadow-indigo-500/20 disabled:opacity-70 mt-6 flex justify-center items-center cursor-pointer text-xs uppercase tracking-wider"
          >
            {loading ? (
               <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : "Submit Registration"}
          </button>

          <p className="text-center text-xs text-slate-500 font-semibold mt-4">
            Already have an account? <a href="/login" className="text-indigo-400 font-bold hover:underline">Sign in</a>
          </p>
        </form>
      </div>
    </div>
  );
}
