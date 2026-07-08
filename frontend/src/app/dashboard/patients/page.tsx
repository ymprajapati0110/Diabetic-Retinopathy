"use client";

import { useState, useEffect } from 'react';
import { Users, Search, Plus, Activity, ChevronRight } from 'lucide-react';
import Link from 'next/link';
import api from '@/lib/api';

export default function PatientsPage() {
  const [patients, setPatients] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const res = await api.get('/patients/');
        // Sort by newest first assuming _id mapping or just reverse
        setPatients(res.data.reverse());
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchPatients();
  }, []);

  const filtered = patients.filter(p => 
    p.patient_reference_code?.toLowerCase().includes(query.toLowerCase()) || 
    p.medical_history?.diabetes_type?.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="space-y-6 flex flex-col h-full">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-4 border-b border-slate-200 pb-5">
         <div>
           <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Patient Directory</h2>
           <p className="mt-2 text-slate-500 font-medium">Manage and review your clinic's patient records.</p>
         </div>
         <div className="mt-4 sm:mt-0 flex items-center gap-3 w-full sm:w-auto">
            <div className="relative w-full sm:w-64">
               <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
               <input 
                 type="text" 
                 placeholder="Search ID or Type..." 
                 className="w-full pl-10 pr-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm focus:ring-2 focus:ring-indigo-500 outline-none shadow-sm"
                 value={query}
                 onChange={e => setQuery(e.target.value)}
               />
            </div>
            <Link href="/dashboard?new=true" className="whitespace-nowrap flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2.5 rounded-xl font-bold transition-all shadow-md shadow-indigo-200">
               <Plus size={18} /> <span className="hidden sm:inline">New Patient</span>
            </Link>
         </div>
      </div>

      <div className="flex-1 bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden relative flex flex-col min-h-[500px]">
          {loading ? (
             <div className="flex-1 flex flex-col items-center justify-center p-12 text-slate-500">
                <div className="w-10 h-10 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4" />
                <span className="font-bold text-sm tracking-widest uppercase">Fetching Records...</span>
             </div>
          ) : patients.length === 0 ? (
             <div className="flex-1 flex flex-col items-center justify-center p-12 text-slate-500">
                <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-6">
                   <Users size={32} className="text-slate-300" />
                </div>
                <h4 className="text-xl font-bold text-slate-900 mb-2">No Patients Enrolled</h4>
                <p className="max-w-sm text-center font-medium mb-6">Add your first clinical patient to begin running AI fundus diagnostics.</p>
                <Link href="/dashboard?new=true" className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-bold transition-all shadow-lg shadow-indigo-200/50 flex items-center gap-2">
                   <Plus size={20} /> Register Patient
                </Link>
             </div>
          ) : (
            <div className="flex-1 overflow-auto p-4 md:p-6 bg-slate-50/50">
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                 {filtered.map((patient, i) => (
                   <div key={patient.id}>
                     <Link href={`/dashboard/patients/${patient.id}`} className="block group h-full">
                       <div className="bg-white hover:bg-slate-50 p-6 rounded-3xl border border-slate-200 hover:border-indigo-300 transition-all shadow-sm hover:shadow-md h-full relative overflow-hidden flex flex-col">
                          {/* Active Indicator Line */}
                          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-transparent group-hover:bg-indigo-500 transition-colors" />
                          
                          <div className="flex items-start justify-between mb-6">
                             <div>
                                <h3 className="font-extrabold text-xl text-slate-900 tracking-tight group-hover:text-indigo-700 transition-colors">{patient.patient_reference_code}</h3>
                                <p className="text-xs font-bold text-slate-500 mt-1.5 uppercase tracking-wider bg-slate-100 inline-block px-2 py-0.5 rounded-full">{patient.age} Y/O • <span className="capitalize">{patient.gender}</span></p>
                             </div>
                             <div className="w-10 h-10 bg-slate-50 border border-slate-100 rounded-full shadow-sm flex items-center justify-center text-slate-400 group-hover:text-indigo-600 group-hover:bg-indigo-50 transition-all shrink-0">
                                <ChevronRight size={20} />
                             </div>
                          </div>

                          <div className="mt-auto pt-4 border-t border-slate-100 flex items-center gap-4">
                             <div className="flex items-center gap-2 text-sm text-slate-700 font-semibold bg-white border border-slate-200 px-3 py-1.5 rounded-xl shadow-sm">
                                <Activity size={16} className={patient.medical_history?.diabetes_type === 'type_1' ? "text-orange-500" : patient.medical_history?.diabetes_type === 'type_2' ? "text-blue-500" : "text-slate-400"} />
                                <span>{patient.medical_history?.diabetes_type === 'type_1' ? "Type 1" : patient.medical_history?.diabetes_type === 'type_2' ? "Type 2" : "Unknown"}</span>
                             </div>
                             <div className="text-sm text-slate-500 font-bold px-3 py-1.5 border border-slate-200 rounded-xl bg-white shadow-sm flex-1 text-center truncate">
                               {patient.medical_history?.duration_years || 0} Yrs Duration
                             </div>
                          </div>
                          
                       </div>
                     </Link>
                   </div>
                 ))}
              </div>
              {filtered.length === 0 && (
                <div className="text-center py-16 text-slate-500 font-medium">
                   <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                     <Search className="text-slate-300" size={24} />
                   </div>
                   No patients match your search criteria.
                </div>
              )}
            </div>
          )}
      </div>
    </div>
  );
}
