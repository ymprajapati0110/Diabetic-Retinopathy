"use client";

import { useState, useEffect } from 'react';
import { Users, FileSearch, ArrowRight, Activity, Clock, PlusCircle, X } from 'lucide-react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import api from '@/lib/api';

function NewPatientModal({ isOpen, onClose, onSuccess }: { isOpen: boolean, onClose: () => void, onSuccess: () => void }) {
  const [formData, setFormData] = useState({ age: '', gender: 'Male', diabetes_type: 'Type 2', duration_years: '' });
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await api.post('/patients/', {
        age: parseInt(formData.age),
        gender: formData.gender,
        medical_history: {
          diabetes_type: formData.diabetes_type,
          duration_years: parseInt(formData.duration_years) || 0
        }
      });
      onSuccess();
      onClose();
    } catch (err) {
      console.error(err);
      alert("Failed to add patient.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" onClick={onClose}></div>
      <div 
        className="bg-white rounded-3xl shadow-2xl w-full max-w-md relative z-10 overflow-hidden"
      >
        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50">
          <h3 className="font-bold text-slate-900">Add New Patient</h3>
          <button onClick={onClose} className="p-1 text-slate-400 hover:bg-slate-200 rounded-lg transition-colors">
            <X size={20} />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-500 uppercase mb-1">Age</label>
              <input type="number" required min="0" max="120" value={formData.age} onChange={e => setFormData({...formData, age: e.target.value})} className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-500 uppercase mb-1">Gender</label>
              <select value={formData.gender} onChange={e => setFormData({...formData, gender: e.target.value})} className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none">
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-500 uppercase mb-1">Diabetes Type</label>
            <select value={formData.diabetes_type} onChange={e => setFormData({...formData, diabetes_type: e.target.value})} className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none">
              <option>Type 1</option>
              <option>Type 2</option>
              <option>Gestational</option>
              <option>Unknown</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-500 uppercase mb-1">Duration of Diabetes (Years)</label>
            <input type="number" required min="0" value={formData.duration_years} onChange={e => setFormData({...formData, duration_years: e.target.value})} className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none" />
          </div>
          <div className="pt-4 flex justify-end gap-3">
            <button type="button" onClick={onClose} className="px-4 py-2 font-medium text-slate-500 hover:bg-slate-100 rounded-xl transition-colors">Cancel</button>
            <button type="submit" disabled={loading} className="px-5 py-2 font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-xl transition-colors disabled:opacity-50">
              {loading ? "Saving..." : "Add Patient"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [patients, setPatients] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const searchParams = useSearchParams();
  const isModalOpen = searchParams.get('new') === 'true';

  const fetchPatients = async () => {
    try {
      const res = await api.get('/patients/');
      setPatients(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  return (
    <div className="space-y-6">
      
        {isModalOpen && (
          <NewPatientModal 
            isOpen={isModalOpen} 
            onClose={() => router.push('/dashboard')} 
            onSuccess={() => { fetchPatients(); router.push('/dashboard'); }} 
          />
        )}
      

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-8 border-b border-slate-200 pb-5">
         <div>
           <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">OcularAI Dashboard</h2>
           <p className="mt-2 text-slate-500 font-medium">Overview of your assigned patient scans and AI diagnostics.</p>
         </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {[
          { label: "Total Patients", value: patients.length.toString(), icon: Users, color: "blue" },
          { label: "Pending Scans", value: "3", icon: Clock, color: "amber" },
          { label: "Urgent Review", value: "1", icon: Activity, color: "red" },
          { label: "Completed AI", value: "12", icon: FileSearch, color: "indigo" },
        ].map((stat, i) => (
          <div 
            key={i} 
            className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 hover:shadow-md transition-shadow relative overflow-hidden group"
          >
            <div className={`absolute -right-6 -top-6 w-24 h-24 bg-${stat.color}-500/10 rounded-full blur-2xl group-hover:bg-${stat.color}-500/20 transition-colors`}></div>
            <div className="flex items-center justify-between mb-4 relative z-10">
              <span className="text-slate-500 text-sm font-semibold uppercase tracking-wider">{stat.label}</span>
              <div className={`w-10 h-10 rounded-xl bg-${stat.color}-50 flex items-center justify-center text-${stat.color}-600`}>
                 <stat.icon size={20} />
              </div>
            </div>
            <div className="text-3xl font-extrabold text-slate-900">{stat.value}</div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden relative">
        <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Users size={20} className="text-indigo-600" /> Patient Registry
          </h3>
          <Link href="/dashboard?new=true" className="text-sm font-medium text-indigo-600 hover:text-indigo-700 bg-indigo-50 px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-colors">
            <PlusCircle size={16} /> Add 
          </Link>
        </div>

        <div className="overflow-x-auto">
          {loading ? (
             <div className="p-8 text-center text-slate-500 animate-pulse flex flex-col items-center">
                <div className="w-8 h-8 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4" />
                Loading registry data...
             </div>
          ) : patients.length === 0 ? (
             <div className="p-12 text-center text-slate-500 py-16">
               <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-4 text-slate-400">
                  <FileSearch size={32} />
               </div>
               <h4 className="text-lg font-semibold text-slate-900 mb-1">No Patients Found</h4>
               <p className="max-w-sm mx-auto">Get started by adding your first patient to the AI registry.</p>
             </div>
          ) : (
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50/80">
                <tr>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Patient ID</th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">Demographics</th>
                  <th scope="col" className="px-6 py-4 text-left text-xs font-bold text-slate-500 uppercase tracking-wider">History</th>
                  <th scope="col" className="px-6 py-4 text-right text-xs font-bold text-slate-500 uppercase tracking-wider">Action</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-slate-100">
                {patients.map((p, i) => (
                  <tr 
                    key={p.id} 
                    className="hover:bg-slate-50/80 transition-colors group cursor-pointer"
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-indigo-600 group-hover:text-indigo-700 transition-colors">
                      {p.patient_reference_code}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-semibold text-slate-900">{p.age} years / {p.gender}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                      {p.medical_history?.diabetes_type || 'Unknown'} - {p.medical_history?.duration_years || 0} yrs
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                      <Link href={`/dashboard/patients/${p.id}`} className="inline-flex items-center gap-1.5 text-indigo-600 hover:text-indigo-900 font-semibold bg-indigo-50 px-3 py-1.5 rounded-lg transition-all">
                        View Scans <ArrowRight size={16} />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
