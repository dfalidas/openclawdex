import { contextBridge, ipcRenderer } from "electron";
import type { SessionInfo, HistoryMessage, ProjectInfo } from "@openclawdex/shared";

contextBridge.exposeInMainWorld("openclawdex", {
  platform: process.platform,

  /** Check which agent CLIs are available on this machine. */
  checkClaude: (): Promise<{ available: boolean; geminiAvailable?: boolean }> =>
    ipcRenderer.invoke("claude:check"),

  /** Send a user message to an agent for a given thread. */
  send: (
    threadId: string,
    message: string,
    opts?: {
      provider?: "claude" | "gemini";
      resumeSessionId?: string;
      projectId?: string;
      images?: { name: string; base64: string; mediaType: string }[];
    },
  ): Promise<void> => ipcRenderer.invoke("claude:send", threadId, message, opts),

  /** Interrupt the current turn for a thread. */
  interrupt: (threadId: string, provider?: "claude" | "gemini"): Promise<void> =>
    ipcRenderer.invoke("claude:interrupt", threadId, provider),

  /** Respond to a deferred tool call (Claude only). */
  respondToTool: (threadId: string, toolUseId: string, responseText: string, provider?: "claude" | "gemini"): Promise<void> =>
    ipcRenderer.invoke("claude:respond-to-tool", threadId, toolUseId, responseText, provider),

  /** List all persisted sessions across all projects. */
  listSessions: (): Promise<SessionInfo[]> =>
    ipcRenderer.invoke("claude:list-sessions"),

  /** Load message history for a session by its session ID. */
  loadHistory: (sessionId: string, provider?: "claude" | "gemini"): Promise<HistoryMessage[]> =>
    ipcRenderer.invoke("claude:load-history", sessionId, provider),

  // ── Projects ────────────────────────────────────────────────

  createProject: (): Promise<ProjectInfo | null> =>
    ipcRenderer.invoke("projects:create"),

  listProjects: (): Promise<ProjectInfo[]> =>
    ipcRenderer.invoke("projects:list"),

  renameProject: (projectId: string, name: string): Promise<void> =>
    ipcRenderer.invoke("projects:rename", projectId, name),

  deleteProject: (projectId: string): Promise<void> =>
    ipcRenderer.invoke("projects:delete", projectId),

  addFolder: (projectId: string, folderPath: string): Promise<string> =>
    ipcRenderer.invoke("projects:add-folder", projectId, folderPath),

  removeFolder: (folderId: string): Promise<void> =>
    ipcRenderer.invoke("projects:remove-folder", folderId),

  // ── Git ─────────────────────────────────────────────────────

  getGitBranch: (cwd: string): Promise<string | null> =>
    ipcRenderer.invoke("git:branch", cwd),

  // ── Threads ─────────────────────────────────────────────────

  renameThread: (sessionId: string, name: string): Promise<void> =>
    ipcRenderer.invoke("threads:rename", sessionId, name),

  deleteThread: (sessionId: string): Promise<void> =>
    ipcRenderer.invoke("threads:delete", sessionId),

  onEvent: (callback: (event: unknown) => void): (() => void) => {
    const handler = (_ipc: Electron.IpcRendererEvent, event: unknown) => callback(event);
    ipcRenderer.on("claude:event", handler);
    return () => {
      ipcRenderer.removeListener("claude:event", handler);
    };
  },
});
