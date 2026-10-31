import { useState, useEffect } from 'react'
import { Sun, Moon, Monitor, Save, Trash2, Wifi, WifiOff, TestTube, Plus, Check, X, ExternalLink, Database } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Input } from '@/components/ui/Input'
import { Dialog } from '@/components/ui/Dialog'
import { Tooltip } from '@/components/ui/Tooltip'
import { Separator } from '@/components/ui/Separator'
import { Label } from '@/components/ui/Label'
import { PageLayout } from '@/components/layout'
import { useTheme } from '@/features/backend-connection/ThemeContext'
import { useBackend } from '@/features/backend-connection/BackendContext'
import { cn } from '@/lib/utils'

export function SettingsPage() {
  const { theme, resolvedTheme, setTheme } = useTheme()
  const { 
    backendUrl, 
    health, 
    isConnected, 
    isConnecting, 
    connectionError, 
    profiles, 
    activeProfile, 
    testConnection, 
    addProfile, 
    removeProfile, 
    setActiveProfile 
  } = useBackend()
  
  const [newProfileUrl, setNewProfileUrl] = useState('')
  const [newProfileName, setNewProfileName] = useState('')
  const [newProfileType, setNewProfileType] = useState<'local' | 'colab' | 'cloud'>('local')
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [testingUrl, setTestingUrl] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<'success' | 'error' | null>(null)

  const handleAddProfile = async () => {
    if (!newProfileUrl.trim() || !newProfileName.trim()) return
    addProfile({
      name: newProfileName,
      url: newProfileUrl.trim(),
      type: newProfileType,
    })
    setNewProfileUrl('')
    setNewProfileName('')
    setAddDialogOpen(false)
  }

  const handleTestConnection = async (url: string) => {
    setTestingUrl(url)
    setTestResult(null)
    try {
      const success = await testConnection(url)
      setTestResult(success ? 'success' : 'error')
    } catch {
      setTestResult('error')
    }
    setTestingUrl(null)
  }

  const formatUrl = (url: string) => {
    try {
      return new URL(url).host
    } catch {
      return url
    }
  }

  return (
    <PageLayout title="Settings" description="Theme, backend connections, and preferences">
      {/* Theme Section */}
      <Card variant="elevated" className="mb-8">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Appearance</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-3">
            {[
              { value: 'light', label: 'Light', icon: Sun, desc: 'Always light mode' },
              { value: 'dark', label: 'Dark', icon: Moon, desc: 'Always dark mode' },
              { value: 'system', label: 'System', icon: Monitor, desc: 'Follow system preference' },
            ].map(({ value, label, icon: Icon, desc }) => (
              <Button
                key={value}
                variant={theme === value ? 'primary' : 'secondary'}
                className="h-24 flex-col gap-2 text-left p-4"
                onClick={() => setTheme(value as 'light' | 'dark' | 'system')}
              >
                <div className="flex items-center gap-2">
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{label}</span>
                  {theme === value && <Check className="h-4 w-4 ml-auto" />}
                </div>
                <p className="text-xs text-[var(--fg-muted)]">{desc}</p>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Backend Section */}
      <Card variant="elevated" className="mb-8">
        <CardHeader>
          <CardTitle>Backend Connection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Current connection status */}
          <div className="p-4 rounded-md border border-[var(--border)] bg-[var(--raised)]/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={cn('w-3 h-3 rounded-full', isConnected ? 'bg-green-500' : 'bg-[var(--danger)]')} />
                <div>
                  <p className="font-medium text-[var(--fg)]">
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </p>
                  <p className="text-sm text-[var(--fg-muted)] font-mono">
                    {formatUrl(backendUrl)} • {health?.device?.toUpperCase() || 'Unknown'}
                  </p>
                </div>
              </div>
              {isConnecting ? (
                <Button variant="secondary" disabled className="gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" /></svg>
                  Testing...
                </Button>
              ) : (
                <Button 
                  variant="secondary" 
                  size="sm" 
                  onClick={() => handleTestConnection(backendUrl)}
                  className="gap-2"
                >
                  <Wifi className="h-4 w-4" />
                  Test Connection
                </Button>
              )}
            </div>
            {connectionError && (
              <p className="mt-2 text-sm text-[var(--danger)]">{connectionError}</p>
            )}
            {testResult === 'success' && !isConnecting && (
              <p className="mt-2 text-sm text-green-500">Connection successful!</p>
            )}
            {testResult === 'error' && !isConnecting && (
              <p className="mt-2 text-sm text-[var(--danger)]">Connection failed</p>
            )}
          </div>

          {/* Saved profiles */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-medium text-[var(--fg)]">Saved Profiles</h4>
              <Button variant="secondary" size="sm" onClick={() => setAddDialogOpen(true)} className="gap-1">
                <Plus className="h-3 w-3" /> Add Profile
              </Button>
            </div>

            {profiles.length === 0 ? (
              <div className="text-center py-8 text-[var(--fg-muted)]">
                <WifiOff className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No backend profiles saved</p>
                <p className="text-sm mt-1">Add a profile to quickly switch between backends</p>
              </div>
            ) : (
              <div className="space-y-2">
                {profiles.map(profile => (
                  <div key={profile.id} className={cn(
                    'flex items-center justify-between p-3 rounded-md border transition-colors',
                    activeProfile?.id === profile.id 
                      ? 'border-[var(--primary)] bg-[var(--primary)]/5' 
                      : 'border-[var(--border)] hover:border-[var(--border-strong)]'
                  )}>
                    <div className="flex items-center gap-3">
                      {profile.type === 'local' && <Monitor className="h-5 w-5 text-[var(--fg-muted)]" />}
                      {profile.type === 'colab' && <TestTube className="h-5 w-5 text-[var(--primary)]" />}
                      {profile.type === 'cloud' && <ExternalLink className="h-5 w-5 text-[var(--missing)]" />}
                      <div>
                        <p className="font-medium text-[var(--fg)]">{profile.name}</p>
                        <p className="text-xs text-[var(--fg-muted)] font-mono">{formatUrl(profile.url)}</p>
                      </div>
                      {activeProfile?.id === profile.id && (
                        <Badge variant="primary" size="sm">Active</Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        onClick={() => handleTestConnection(profile.url)}
                        disabled={isConnecting}
                        className="h-8 w-8"
                      >
                        <Wifi className="h-4 w-4" />
                      </Button>
                      <Button 
                        variant={activeProfile?.id === profile.id ? 'secondary' : 'ghost'} 
                        size="sm" 
                        onClick={() => setActiveProfile(profile.id)}
                        className="gap-1"
                      >
                        {activeProfile?.id === profile.id ? <Check className="h-3 w-3" /> : 'Activate'}
                      </Button>
                      {profiles.length > 1 && (
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          onClick={() => removeProfile(profile.id)}
                          className="h-8 w-8 text-[var(--danger)] hover:bg-[var(--danger)]/10"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Add profile dialog */}
          <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen} title="Add Backend Profile">
            <div className="space-y-4">
              <Input
                label="Profile Name"
                placeholder="e.g., Local GPU, Colab, Cloud Run"
                value={newProfileName}
                onChange={e => setNewProfileName(e.target.value)}
              />
              <Input
                label="Backend URL"
                placeholder="http://localhost:8000 or https://xxx.ngrok.io"
                value={newProfileUrl}
                onChange={e => setNewProfileUrl(e.target.value)}
              />
              <div>
                <Label>Profile Type</Label>
                <div className="grid grid-cols-3 gap-2 mt-2">
                  {[
                    { value: 'local', label: 'Local', icon: Monitor },
                    { value: 'colab', label: 'Colab', icon: TestTube },
                    { value: 'cloud', label: 'Cloud', icon: ExternalLink },
                  ].map(({ value, label, icon: Icon }) => (
                    <Button
                      key={value}
                      variant={newProfileType === value ? 'primary' : 'secondary'}
                      className="flex flex-col items-center gap-1 py-3"
                      onClick={() => setNewProfileType(value as any)}
                    >
                      <Icon className="h-5 w-5" />
                      <span className="text-xs">{label}</span>
                    </Button>
                  ))}
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-4">
                <Button variant="secondary" onClick={() => setAddDialogOpen(false)}>Cancel</Button>
                <Button onClick={handleAddProfile} disabled={!newProfileUrl.trim() || !newProfileName.trim()}>
                  Save Profile
                </Button>
              </div>
            </div>
          </Dialog>
        </CardContent>
      </Card>

      {/* Data & Privacy */}
      <Card variant="elevated" className="mb-8">
        <CardHeader>
          <CardTitle>Data & Privacy</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 rounded-md border border-[var(--border)] bg-[var(--raised)]/50">
            <div className="flex items-start gap-3">
              <Database className="h-5 w-5 text-[var(--fg-muted)] flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-medium text-[var(--fg)]">Local Storage Only</h4>
                <p className="text-sm text-[var(--fg-muted)] mt-1">
                  All settings (theme, backend profiles) are stored locally in your browser.
                  No data is sent to external servers except your configured backend.
                </p>
              </div>
            </div>
          </div>
          
          <div className="p-4 rounded-md border border-[var(--border)] bg-[var(--raised)]/50">
            <div className="flex items-start gap-3">
              <WifiOff className="h-5 w-5 text-[var(--fg-muted)] flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-medium text-[var(--fg)]">No Authentication</h4>
                <p className="text-sm text-[var(--fg-muted)] mt-1">
                  This research platform does not implement authentication.
                  Secure your backend with network-level controls (VPN, firewall, etc.).
                </p>
              </div>
            </div>
          </div>
          
          <div className="p-4 rounded-md border border-[var(--border)] bg-[var(--raised)]/50">
            <div className="flex items-start gap-3">
              <Save className="h-5 w-5 text-[var(--fg-muted)] flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-medium text-[var(--fg)]">Experiment Artifacts</h4>
                <p className="text-sm text-[var(--fg-muted)] mt-1">
                  Reconstruction runs are saved to <code className="bg-[var(--bg)] px-1 rounded font-mono">outputs/runs/</code> on the backend server.
                  Manage via the Experiment History page.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* About */}
      <Card variant="elevated">
        <CardHeader>
          <CardTitle>About</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-[var(--fg-muted)]">Platform</span>
              <span className="font-mono text-[var(--fg)]">Audio Inpainting Research Platform</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--fg-muted)]">Version</span>
              <span className="font-mono text-[var(--fg)]">1.0.0 (Research)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--fg-muted)]">Paper</span>
              <span className="font-mono text-[var(--fg)]">Miotello et al., TASLP 2024</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--fg-muted)]">DOI</span>
              <a href="https://doi.org/10.1109/TASLP.2023.3324556" target="_blank" rel="noopener noreferrer" className="font-mono text-[var(--primary)] hover:underline">
                10.1109/TASLP.2023.3324556
              </a>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--fg-muted)]">Repository</span>
              <a href="https://github.com/charan-0845/Audio_Inpainting" target="_blank" rel="noopener noreferrer" className="font-mono text-[var(--primary)] hover:underline flex items-center gap-1">
                charan-0845/Audio_Inpainting
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </div>
        </CardContent>
      </Card>
    </PageLayout>
  )
}