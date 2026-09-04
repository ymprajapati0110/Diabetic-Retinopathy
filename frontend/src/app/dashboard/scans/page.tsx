"use client";

import { FileSearch, Activity, Calendar } from 'lucide-react';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import api from '@/lib/api';

export default function ScansPage() {
  const [scans, setScans] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchScans = async () => {
      try {
        // We'll fetch all patients first, then all scans for those patients, or just create a new endpoint
        // For now, since we don't have a /scans/doctor/me endpoint, we can use the patients to fetch scans
        // But since this is just a UI implementation task, let's fetch patients and their scans
        const patientsRes = await api.get('/patients/');
        const allScans = [];
        
        for (const pt of patientsRes.data) {
          const scansRes = await api.get(`/scans/patient/${pt.id}`);
          for (const s of scansRes.data) {
             allScans.push({...s, patient: pt});
          }
        }
        
        setScans(allScans.sort((a,b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
      } catch (err) {
        console.error("Failed to fetch scans:", err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchScans();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 border-b border-slate-200 pb-5">
         <div>
           <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Recent Scans</h2>
           <p className="mt-2 text-slate-500 font-medium">All historical AI diagnostics mapped across your patients.</p>
         </div>
      </div>

      <div className="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden relative p-8">
        {loading ? (
             <div className="p-8 text-center text-slate-500 animate-pulse flex flex-col items-center">
                <div className="w-8 h-8 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4" />
                Aggregating scan database...
             </div>
        ) : scans.length === 0 ? (
             <div className="p-12 text-center text-slate-500 py-16">
               <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-4 text-slate-400">
                  <FileSearch size={32} />
               </div>
               <h4 className="text-lg font-semibold text-slate-900 mb-1">No Scans Found</h4>
               <p className="max-w-sm mx-auto">Upload a fundus image from a patient profile to generate a scan.</p>
             </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {scans.map((scan, i) => (
              <div 
                key={i}
                className="bg-slate-50 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow overflow-hidden group"
              >
                <Link href={`/dashboard/patients/${scan.patient_id}`}>
                  <div className="aspect-video bg-slate-900 relative overflow-hidden flex items-center justify-center">
                     <img src={scan.raw_image_s3_url} className="absolute inset-0 w-full h-full object-cover opacity-70 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500" />
                     {scan.status !== 'completed' && <Activity className="absolute text-white animate-pulse" size={32} />}
                     <div className="absolute top-3 left-3 bg-slate-900/80 backdrop-blur-md px-2 py-1 rounded border border-slate-700 text-xs text-white font-mono flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${scan.status === 'completed' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]' : 'bg-yellow-500 animate-pulse'}`}></span>
                        {scan.status.toUpperCase()}
                     </div>
                  </div>
                </Link>
                <div className="p-5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-bold text-slate-900 truncate">Patient: {scan.patient?.patient_reference_code || scan.patient_id}</span>
                    <span className="bg-indigo-50 text-indigo-700 text-xs px-2 py-0.5 rounded font-bold uppercase">{scan.eye_side}</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-500 font-medium mb-4">
                    <Calendar size={14} /> {new Date(scan.created_at).toLocaleString()}
                  </div>
                  {scan.dr_prediction_level !== null && scan.dr_prediction_level !== undefined && (
                    <div className="flex items-center justify-between text-sm px-3 py-2 bg-white rounded-xl border border-slate-100">
                      <span className="text-slate-500 font-semibold">AI Severity:</span>
                      <span className="font-bold text-slate-900">Level {scan.dr_prediction_level}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
