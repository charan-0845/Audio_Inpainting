import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from '@/components/layout'
import { Suspense, lazy } from 'react'

// Lazy load all page components for code-splitting and to prevent tree-shaking
const LandingPage = lazy(() => import('@/pages/LandingPage').then(m => ({ default: m.LandingPage })))
const DashboardPage = lazy(() => import('@/pages/DashboardPage').then(m => ({ default: m.DashboardPage })))
const WorkspacePage = lazy(() => import('@/pages/WorkspacePage').then(m => ({ default: m.WorkspacePage })))
const ExperimentsPage = lazy(() => import('@/pages/ExperimentsPage').then(m => ({ default: m.ExperimentsPage })))
const ExperimentResultsPage = lazy(() => import('@/pages/ExperimentResultsPage').then(m => ({ default: m.ExperimentResultsPage })))
const CompareExperimentsPage = lazy(() => import('@/pages/CompareExperimentsPage').then(m => ({ default: m.CompareExperimentsPage })))
const ResearchPage = lazy(() => import('@/pages/ResearchPage').then(m => ({ default: m.ResearchPage })))
const MethodologyPage = lazy(() => import('@/pages/MethodologyPage').then(m => ({ default: m.MethodologyPage })))
const SettingsPage = lazy(() => import('@/pages/SettingsPage').then(m => ({ default: m.SettingsPage })))

function PageWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<div className="flex h-64 items-center justify-center"><div className="text-[var(--fg-muted)]">Loading...</div></div>}>
      {children}
    </Suspense>
  )
}

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout isLanding><PageWrapper><LandingPage /></PageWrapper></Layout>} />
      <Route path="/dashboard" element={<Layout><PageWrapper><DashboardPage /></PageWrapper></Layout>} />
      <Route path="/workspace/new" element={<Layout><PageWrapper><WorkspacePage /></PageWrapper></Layout>} />
      <Route path="/experiments" element={<Layout><PageWrapper><ExperimentsPage /></PageWrapper></Layout>} />
      <Route path="/experiments/:id" element={<Layout><PageWrapper><ExperimentResultsPage /></PageWrapper></Layout>} />
      <Route path="/experiments/compare" element={<Layout><PageWrapper><CompareExperimentsPage /></PageWrapper></Layout>} />
      <Route path="/research" element={<Layout isLanding><PageWrapper><ResearchPage /></PageWrapper></Layout>} />
      <Route path="/research/methodology" element={<Layout isLanding><PageWrapper><MethodologyPage /></PageWrapper></Layout>} />
      <Route path="/settings" element={<Layout><PageWrapper><SettingsPage /></PageWrapper></Layout>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}