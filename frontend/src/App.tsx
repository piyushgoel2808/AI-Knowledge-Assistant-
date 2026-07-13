import { useEffect, useMemo, useState } from 'react'
import './styles.css'
import { askQuestion, clearDocuments, getHealth, uploadDocuments, type ChatMessage, type Source } from './api'

type Status = {
  provider: string
  stored_chunks: number
  status: string
}

type ChatItem = ChatMessage & { sources?: Source[] }

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [history, setHistory] = useState<ChatItem[]>([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    getHealth().then(setStatus).catch((error: unknown) => setMessage(error instanceof Error ? error.message : 'Failed to load status'))
  }, [])

  const documentCountLabel = useMemo(() => `${status?.stored_chunks ?? 0} chunks indexed`, [status])

  function handleFileSelection(event: React.ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(event.target.files ?? []))
  }

  async function handleUpload() {
    if (!files.length) {
      setMessage('Choose one or more PDF, DOCX, or TXT files first.')
      return
    }
    setBusy(true)
    setMessage(null)
    try {
      const result = await uploadDocuments(files)
      setMessage(`${result.message} ${result.chunks_indexed} chunks added.`)
      setStatus(await getHealth())
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  async function handleAsk() {
    const trimmed = question.trim()
    if (!trimmed) return
    setBusy(true)
    setMessage(null)
    const userMessage: ChatItem = { role: 'user', content: trimmed }
    const nextHistory = [...history, userMessage]
    setHistory(nextHistory)
    setQuestion('')
    try {
      const result = await askQuestion(trimmed, history)
      setHistory([...nextHistory, { role: 'assistant', content: result.answer, sources: result.sources }])
      setMessage(result.refusal ? 'The answer was not found in the uploaded documents.' : null)
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : 'Chat request failed')
    } finally {
      setBusy(false)
    }
  }

  async function handleClear() {
    setBusy(true)
    setMessage(null)
    try {
      await clearDocuments()
      setHistory([])
      setFiles([])
      setStatus(await getHealth())
      setMessage('Documents cleared.')
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : 'Clear failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-card">
          <p className="eyebrow">AI Knowledge Assistant</p>
          <h1>AI Knowledge Assistant Q&A</h1>
          <p className="muted">Document Uploading,and asking questions from the uploaded documents.</p>
        </div>

        <section className="panel">
          <h2>Workspace</h2>
          <div className="stat-grid">
            <div>
              <span>Provider</span>
              <strong>{status?.provider ?? '...'}</strong>
            </div>
            <div>
              <span>Index</span>
              <strong>{documentCountLabel}</strong>
            </div>
          </div>
          <label className="file-input">
            <input type="file" multiple accept=".pdf,.docx,.txt" onChange={handleFileSelection} />
            <span>{files.length ? `${files.length} file(s) selected` : 'Choose PDF, DOCX, or TXT files'}</span>
          </label>
          <div className="button-row">
            <button onClick={handleUpload} disabled={busy}>Upload & Index</button>
            <button className="secondary" onClick={handleClear} disabled={busy}>Clear Docs</button>
          </div>
        </section>

        <section className="panel compact">
          <h2>How it works</h2>
          <ul>
            <li>Files are indexed locally in Chroma.</li>
            <li>Answers come only from uploaded documents.</li>
            <li>Each answer includes sources</li>
          </ul>
        </section>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">Conversation</p>
            <h2>Ask about your documents</h2>
          </div>
          {message ? <div className="toast">{message}</div> : null}
        </header>

        <section className="chat-card">
          <div className="chat-thread">
            {history.length === 0 ? (
              <div className="empty-state">
                <h3>No messages yet</h3>
                <p>Document Uploading,and asking questions from the uploaded documents.</p>
              </div>
            ) : null}

            {history.map((item, index) => (
              <article key={`${item.role}-${index}`} className={`message ${item.role}`}>
                <div className="message-meta">{item.role}</div>
                <p>{item.content}</p>
                {item.sources?.length ? (
                  <details>
                    <summary>View sources</summary>
                    <div className="source-list">
                      {item.sources.map((source, sourceIndex) => (
                        <div key={`${source.chunk_id ?? sourceIndex}`} className="source-card">
                          <strong>{source.source}</strong>
                          <span>Page {source.page ?? 'n/a'} · {source.section ?? 'Document'}</span>
                          <pre>{source.snippet}</pre>
                        </div>
                      ))}
                    </div>
                  </details>
                ) : null}
              </article>
            ))}
          </div>

          <div className="composer">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Type your question here..."
              rows={4}
            />
            <div className="button-row">
              <button onClick={handleAsk} disabled={busy}>Send</button>
              <button className="secondary" onClick={() => { setHistory([]); setMessage(null); }} disabled={busy}>Reset Chat</button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
