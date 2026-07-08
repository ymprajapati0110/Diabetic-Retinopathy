"use client";

import { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { Home, Users, Settings, LogOut, FileSearch, Stethoscope, Bell, Menu, X, Scan } from 'lucide-react';
import Link from 'next/link';
import clsx from 'clsx';
import api from '@/lib/api';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [profile, setProfile] = useState<any>(null);

  useEffect(() => {
    // A quick check if we are logged in - else we could redirect here
    const token = localStorage.getItem('token');
    if (!token) {
      router.push('/login');
    } else {
      // We could fetch profile here if we had an endpoint
      setProfile({ name: "Dr. Smith", role: "verified" });
    }
  }, [router]);

  const navItems = [
    { icon: Home, label: 'Overview', href: '/dashboard' },
    { icon: Users, label: 'Patients', href: '/dashboard/patients' },
    { icon: FileSearch, label: 'Recent Scans', href: '/dashboard/scans' },
    { icon: Scan, label: 'Scan', href: '/dashboard/scan-raw' },
    { icon: Settings, label: 'Settings', href: '/dashboard/settings' },
  ];

  if (!profile) return null; // or loading spinner

  return (
    <div className="flex bg-slate-50 min-h-screen relative overflow-hidden text-slate-900">

      {/* Mobile Overlay */}
      {!isSidebarOpen && (
        <div
          className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setSidebarOpen(true)}
        />
      )}

      {/* Sidebar */}
      <aside
        
        className={clsx(
          "fixed inset-y-0 left-0 z-50 lg:relative bg-white border-r border-slate-200 flex flex-col shadow-2xl lg:shadow-none transition-all duration-300 ease-in-out",
          isSidebarOpen ? "w-72 px-6" : "w-0 px-0 opacity-0 lg:opacity-100 lg:w-20 lg:px-4"
        )}
      >
        <div className="flex items-center gap-3 mb-10 overflow-hidden shrink-0 pt-4">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-indigo-200 shrink-0">
            <Stethoscope size={24} />
          </div>
          <div className={clsx("transition-opacity duration-200", !isSidebarOpen && "lg:hidden")}>
            <h2 className="text-xl font-bold tracking-tight">OcularAI</h2>
            <span className="text-xs text-indigo-600 font-semibold uppercase tracking-wider bg-indigo-50 px-2 py-0.5 rounded-full inline-block mt-0.5">Clinical</span>
          </div>
        </div>

        <nav className="flex-1 space-y-2 overflow-y-auto overflow-x-hidden pt-2 scrollbar-hide">
          {navItems.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
            return (
              <Link key={item.href} href={item.href}
                className={clsx(
                  "flex items-center gap-3 px-3 py-3.5 rounded-xl transition-all group relative overflow-hidden",
                  isActive
                    ? "bg-indigo-50 text-indigo-700 font-medium"
                    : "text-slate-500 hover:bg-slate-50 hover:text-slate-900"
                )}
              >
                {isActive && (
                  <div  className="absolute left-0 w-1 h-8 bg-indigo-600 rounded-r-full top-1/2 -translate-y-1/2" />
                )}
                <item.icon size={20} className={clsx("shrink-0", isActive ? "text-indigo-600" : "text-slate-400 group-hover:text-slate-600")} />
                <span className={clsx("truncate whitespace-nowrap", !isSidebarOpen && "lg:hidden")}>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="mt-8 pt-6 border-t border-slate-100 overflow-hidden relative">
          <div className={clsx("flex items-center gap-3", !isSidebarOpen && "lg:justify-center")}>
            <div className="w-10 h-10 rounded-full bg-slate-900 flex items-center justify-center text-white shrink-0 shadow-md">
              <span className="font-semibold text-sm">DS</span>
            </div>
            <div className={clsx("flex flex-col min-w-0 flex-1 transition-opacity", !isSidebarOpen && "lg:hidden")}>
              <span className="text-sm font-semibold text-slate-900 truncate">{profile.name}</span>
              <span className="text-xs text-slate-500 truncate flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-green-500 shrink-0"></div> {profile.role}
              </span>
            </div>
          </div>
          <button
            onClick={() => { localStorage.removeItem('token'); router.push('/login'); }}
            className={clsx(
              "mt-4 flex items-center gap-2 text-sm text-red-500 hover:bg-red-50 p-2.5 rounded-xl transition-colors w-full",
              !isSidebarOpen && "lg:justify-center"
            )}
          >
            <LogOut size={18} />
            <span className={clsx(!isSidebarOpen && "lg:hidden")}>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden bg-slate-50/50">
        <header className="h-20 bg-white/80 backdrop-blur-md border-b border-slate-200 px-6 lg:px-8 flex items-center justify-between shrink-0 sticky top-0 z-30">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!isSidebarOpen)}
              className="p-2 -ml-2 hover:bg-slate-100 rounded-lg transition-colors text-slate-500"
            >
              <Menu size={24} />
            </button>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-700 hidden sm:block">
              {navItems.find(item => pathname.startsWith(item.href))?.label || 'Dashboard'}
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <button className="relative p-2.5 rounded-full hover:bg-slate-100 text-slate-500 transition-colors">
              <Bell size={20} />
              <span className="absolute top-2 right-2.5 w-2 h-2 bg-red-500 rounded-full ring-2 ring-white"></span>
            </button>
            <Link href="/dashboard?new=true" className="hidden sm:flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2.5 rounded-xl font-medium transition-all shadow-sm shadow-indigo-200 hover:shadow-md hover:shadow-indigo-300">
              <span>New Patient</span>
            </Link>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-6 lg:p-8 scroll-smooth z-10 relative">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#slate-800_1px,transparent_1px),linear-gradient(to_bottom,#slate-800_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-[0.03] pointer-events-none"></div>
          {children}
        </div>
      </main>
    </div>
  );
}
