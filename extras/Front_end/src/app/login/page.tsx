"use client";

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Stethoscope, KeyRound, Mail, AlertCircle, BadgeCheck } from 'lucide-react';
import api from '@/lib/api';

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const isRegistered = searchParams.get('registered') === 'true';

  const [formData, setFormData] = useState({
    username: '', // OAuth2 specifices 'username' for email
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Must use application/x-www-form-urlencoded for OAuth2PasswordRequestForm in FastAPI
      const params = new URLSearchParams();
      params.append('username', formData.username);
      params.append('password', formData.password);

      const res = await api.post('/auth/login', params, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      
      localStorage.setItem('token', res.data.access_token);
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Invalid credentials or account pending verification');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-blue-500/5 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-400/20 via-slate-50 to-slate-50"></div>
      
      <div className="glass w-full max-w-md p-8 rounded-3xl relative z-10">
        <div className="flex flex-col items-center mb-8">
          <div className="w-16 h-16 bg-indigo-100 rounded-2xl flex items-center justify-center mb-4 text-indigo-600 shadow-inner">
            <Stethoscope size={32} />
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">OcularAI Portal</h1>
          <p className="text-slate-500 text-sm mt-1">Doctor Login</p>
        </div>

        {isRegistered && (
          <div className="mb-6 p-4 bg-green-50 text-green-700 rounded-xl text-sm border border-green-100 flex items-start gap-3">
            <BadgeCheck className="shrink-0 mt-0.5" size={18} />
            <div>
              <p className="font-semibold">Registration Successful</p>
              <p className="text-green-600/80 mt-1">Your account is pending verification. An admin will review your medical license shortly.</p>
            </div>
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 bg-red-50 text-red-600 rounded-xl text-sm border border-red-100 flex items-start gap-3 animate-pulse">
            <AlertCircle className="shrink-0 mt-0.5" size={18} />
            <p className="pt-0.5">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider ml-1">Work Email</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input 
                type="email" 
                required
                className="w-full pl-10 pr-4 py-3 bg-white/50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all placeholder:text-slate-400"
                placeholder="doctor@hospital.org"
                value={formData.username}
                onChange={e => setFormData({...formData, username: e.target.value})}
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="flex items-center justify-between text-xs font-semibold text-slate-500 uppercase tracking-wider ml-1">
              <span>Password</span>
              <a href="#" className="font-medium text-indigo-600 hover:text-indigo-700 capitalize text-[11px] hover:underline">Forgot?</a>
            </label>
            <div className="relative">
              <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input 
                type="password" 
                required
                className="w-full pl-10 pr-4 py-3 bg-white/50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all placeholder:text-slate-400 tracking-widest"
                placeholder="••••••••"
                value={formData.password}
                onChange={e => setFormData({...formData, password: e.target.value})}
              />
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full py-3.5 px-4 bg-slate-900 hover:bg-slate-800 text-white rounded-xl font-medium transition-all focus:ring-4 focus:ring-slate-900/20 disabled:opacity-70 mt-4 flex justify-center items-center group"
          >
            {loading ? (
               <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <span className="flex items-center gap-2">
                Secure Login 
                <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
              </span>
            )}
          </button>

          <div className="pt-4 text-center border-t border-slate-100">
            <p className="text-sm text-slate-500">
              New to OcularAI? <a href="/register" className="text-indigo-600 font-semibold hover:underline">Apply for Access</a>
            </p>
          </div>
        </form>
      </div>
    </div>
  );
}



export default function LoginPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">Loading...</div>}>
      <LoginForm />
    </Suspense>
  );
}
