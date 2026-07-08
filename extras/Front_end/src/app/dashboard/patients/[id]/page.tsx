"use client";

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { User, Activity, Clock, ShieldCheck, FileSearch } from 'lucide-react';
import ClinicalViewer from '@/components/ClinicalViewer';
import api from '@/lib/api';

export default function PatientDetailPage() {
  const params = useParams();
  const patientId = params.id as string;
  
  const [patient, setPatient] = useState<any>(null);
  const [scans, setScans] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchPatientData = async () => {
    try {
      const [patientRes, scansRes] = await Promise.all([
        api.get(`/patients/${patientId}`),
        api.get(`/scans/patient/${patientId}`)
      ]);
      setPatient(patientRes.data);
      setScans(scansRes.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatientData();
  }, [patientId]);

  if (loading) return <div className="p-8 text-center animate-pulse text-slate-500 font-bold tracking-widest text-sm uppercase">Loading Record...</div>;
  if (!patient) return <div className="p-8 text-center text-red-500 font-bold bg-red-50 rounded-xl">Record not found.</div>;

  return (
    <div className="space-y-6">
       
      {/* Demographics Card */}
      <div className="bg-white rounded-3xl p-6 lg:p-8 shadow-sm border border-slate-200 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-slate-50 rounded-full blur-3xl -z-10 translate-x-1/2 -translate-y-1/2"></div>
         
        <div className="flex items-center gap-5">
          <div className="w-16 h-16 bg-slate-900 text-white rounded-2xl flex items-center justify-center shadow-md">
            <User size={32} />
          </div>
          <div>
            <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
              {patient.patient_reference_code} 
              <span className="bg-emerald-100 text-emerald-700 text-xs px-2 py-0.5 rounded-full font-bold tracking-wider uppercase flex items-center gap-1">
                <ShieldCheck size={14} /> HIPAA Active
              </span>
            </h2>
            <div className="text-sm font-semibold text-slate-500 mt-1 flex items-center gap-4">
              <span>{patient.age} years old</span>
              <span className="w-1 h-1 bg-slate-300 rounded-full"></span>
              <span className="capitalize">{patient.gender}</span>
            </div>
          </div>
        </div>

        <div className="flex gap-4 w-full md:w-auto overflow-x-auto pb-2 md:pb-0 hide-scrollbar">
           <div className="bg-slate-50 px-4 py-3 rounded-2xl border border-slate-100 flex items-center gap-3 shrink-0">
             <Activity className="text-orange-500" size={20} />
             <div>
               <p className="text-[10px] uppercase font-bold text-slate-400">Diabetes Type</p>
               <p className="font-semibold text-slate-900">{patient.medical_history?.diabetes_type || 'N/A'}</p>
             </div>
           </div>
           <div className="bg-slate-50 px-4 py-3 rounded-2xl border border-slate-100 flex items-center gap-3 shrink-0">
             <Clock className="text-blue-500" size={20} />
             <div>
               <p className="text-[10px] uppercase font-bold text-slate-400">Duration</p>
               <p className="font-semibold text-slate-900">{patient.medical_history?.duration_years || 0} Years</p>
             </div>
           </div>
           <div className="bg-slate-50 px-4 py-3 rounded-2xl border border-slate-100 flex items-center gap-3 shrink-0">
             <FileSearch className="text-indigo-500" size={20} />
             <div>
               <p className="text-[10px] uppercase font-bold text-slate-400">Past Scans</p>
               <p className="font-semibold text-slate-900">{scans.length}</p>
             </div>
           </div>
        </div>
      </div>

      {/* Main Clinical Viewer */}
      <div >
         <ClinicalViewer patientId={patientId} onScanComplete={fetchPatientData} />
      </div>

      {/* Historical Scans (Optional simple list) */}
      {scans.length > 0 && (
         <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-200">
           <h3 className="text-lg font-bold text-slate-900 mb-4 border-b border-slate-100 pb-4">Historical Scans Archive</h3>
           <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
             {scans.map((scan, i) => (
                <div key={i} className="bg-slate-50 rounded-2xl p-4 border border-slate-100 hover:border-slate-300 transition-colors group cursor-pointer relative overflow-hidden">
                  <span className={`absolute top-2 right-2 flex w-2 h-2 rounded-full ${scan.status === 'completed' ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'}`}></span>
                  <div className="aspect-square bg-slate-900 rounded-xl mb-3 overflow-hidden flex items-center justify-center">
                    {scan.status === 'completed' ? (
                      <img src={scan.raw_image_s3_url} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity" alt="Thumb" />
                    ) : (
                      <Activity className="text-slate-700 animate-pulse" size={24} />
                    )}
                  </div>
                  <div className="font-semibold text-xs text-slate-900 capitalize mb-0.5">{scan.eye_side} Eye</div>
                  <div className="text-[10px] text-slate-500 font-mono">{new Date(scan.created_at).toLocaleDateString()}</div>
                </div>
             ))}
           </div>
         </div>
      )}

    </div>
  );
}
