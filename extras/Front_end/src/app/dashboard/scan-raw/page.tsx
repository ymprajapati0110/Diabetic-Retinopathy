"use client";

import { useState, useEffect } from 'react';
import { Scan, UserPlus } from 'lucide-react';
import ClinicalViewer from '@/components/ClinicalViewer';
import api from '@/lib/api';
import Link from 'next/link';

export default function RawScanPage() {
  const [patients, setPatients] = useState<any[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const res = await api.get('/patients/');
        setPatients(res.data);
        if (res.data.length > 0) {
          setSelectedPatientId(res.data[0]._id);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchPatients();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 border-b border-slate-200 pb-5">
         <div>
           <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Raw Image Scan</h2>
           <p className="mt-2 text-slate-500 font-medium">Quickly upload and diagnose fundus images globally.</p>
         </div>
      </div>

      <div className="bg-white rounded-3xl p-6 lg:p-8 shadow-sm border border-slate-200">
        {loading ? (
             <div className="text-slate-500 animate-pulse text-sm font-bold uppercase tracking-widest text-center py-10">Loading Patients...</div>
        ) : patients.length === 0 ? (
             <div className="text-center py-12">
               <UserPlus size={48} className="mx-auto text-slate-300 mb-4" />
               <h3 className="text-lg font-bold text-slate-900 mb-2">No Patients Found</h3>
               <p className="text-slate-500 mb-6 max-w-sm mx-auto">You must have at least one registered patient to assign a diagnostic scan.</p>
               <Link href="/dashboard?new=true" className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-xl font-bold transition-all shadow-md shadow-indigo-200">
                 Register New Patient
               </Link>
             </div>
        ) : (
          <div className="space-y-6">
             <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center p-4 bg-slate-50 rounded-2xl border border-slate-100 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50 rounded-full blur-2xl -mr-10 -mt-10 pointer-events-none"></div>
                <div className="flex-1 w-full relative z-10">
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Select Target Patient</label>
                  <select 
                    className="w-full bg-white border border-slate-200 rounded-xl px-4 py-3 text-sm font-semibold text-slate-900 focus:ring-2 focus:ring-indigo-500 outline-none shadow-sm transition-all cursor-pointer"
                    value={selectedPatientId}
                    onChange={(e) => setSelectedPatientId(e.target.value)}
                  >
                     {patients.map(p => (
                       <option key={p._id} value={p._id}>
                         {p.patient_reference_code} - {p.age}yrs {p.gender}
                       </option>
                     ))}
                  </select>
                </div>
                <div className="px-4 py-3 bg-indigo-50 text-indigo-700 rounded-xl text-sm font-bold border border-indigo-100 shrink-0 self-end sm:self-auto relative z-10">
                   ID: {selectedPatientId.slice(-6).toUpperCase()}
                </div>
             </div>

             <div key={selectedPatientId} className="relative z-10">
                <ClinicalViewer 
                   patientId={selectedPatientId} 
                   onScanComplete={() => {
                     // Could trigger a toast or something, but ClinicalViewer handles local state beautifully
                   }} 
                />
             </div>
          </div>
        )}
      </div>
    </div>
  );
}
