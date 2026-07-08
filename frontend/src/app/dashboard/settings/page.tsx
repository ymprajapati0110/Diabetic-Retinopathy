"use client";

import { Settings, Shield, Bell, User as UserIcon, Lock } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 border-b border-slate-200 pb-5">
         <div>
           <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Portal Settings</h2>
           <p className="mt-2 text-slate-500 font-medium">Manage your security preferences and notifications.</p>
         </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
         {/* Sidebar */}
         <div className="md:col-span-1 space-y-2">
           {[
             { name: "Profile Details", icon: UserIcon, active: true },
             { name: "Security & Login", icon: Shield, active: false },
             { name: "Notifications", icon: Bell, active: false },
             { name: "API Keys (Admin)", icon: Lock, active: false },
           ].map((tab, i) => (
             <button key={i} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${tab.active ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100'}`}>
               <tab.icon size={18} />
               {tab.name}
             </button>
           ))}
         </div>

         {/* Content */}
         <div className="md:col-span-2">
           <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-200 flex flex-col items-center justify-center min-h-[400px] text-center">
              <div className="w-16 h-16 bg-slate-50 text-slate-400 rounded-2xl flex items-center justify-center mb-4">
                 <Settings size={32} />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-2">Settings Hub</h3>
              <p className="text-slate-500 text-sm max-w-xs">Your settings panel is currently locked while your account awaits standard admin privileges assignment.</p>
           </div>
         </div>
      </div>
    </div>
  );
}
