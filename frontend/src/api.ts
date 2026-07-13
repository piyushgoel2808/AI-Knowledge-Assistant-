export type Source = {
  source: string
  page?: number | null
  section?: string | null
  snippet: string
  chunk_id?: string
  distance?: number | null
}

export type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with ${response.status}`)
  }
  return response.json() as Promise<T>
}

export async function getHealth() {
  return parseJsonResponse<{ status: string; provider: string; stored_chunks: number }>(
    await fetch(`${API_URL}/health`),
  )
}

export async function uploadDocuments(files: File[]) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

  return parseJsonResponse<{ message: string; files: string[]; chunks_indexed: number; stored_chunks: number }>(
    await fetch(`${API_URL}/documents/upload`, { method: 'POST', body: formData }),
  )
}

export async function clearDocuments() {
  return parseJsonResponse<{ message: string }>(await fetch(`${API_URL}/documents/clear`, { method: 'POST' }))
}

export async function askQuestion(question: string, history: ChatMessage[]) {
  return parseJsonResponse<{ answer: string; sources: Source[]; confidence: string; refusal: boolean }>(
    await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history }),
    }),
  )
}
