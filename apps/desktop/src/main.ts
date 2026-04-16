import { app, BrowserWindow, ipcMain, nativeTheme, dialog } from "electron";
import { autoUpdater } from "electron-updater";
import { shell } from "electron";
import path from "path";
import { randomUUID } from "crypto";
import { execSync } from "child_process";
import { eq } from "drizzle-orm";
import { findClaudeBinary, ClaudeSession } from "./claude";
import { findGeminiBinary, GeminiSession } from "./gemini";
import { listSessions, getSessionMessages } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { IpcEvent } from "@openclawdex/shared";
import { initDb, getDb } from "./db";
import { knownThreads, projects, projectFolders } from "./db/schema";

const DEV_URL = "http://localhost:3000";
const IS_DEV = !app.isPackaged;
type Provider = "claude" | "gemini";

if (app.isPackaged && process.platform === "darwin") {
  try {
    const path = execSync("/bin/zsh -ilc 'echo $PATH'", { encoding: "utf-8" }).trim();
    if (path) process.env.PATH = path;
  } catch { /* keep default PATH */ }
}

let mainWindow: BrowserWindow | null = null;

const claudePath = findClaudeBinary();
const geminiPath = findGeminiBinary();
const claudeSessions = new Map<string, ClaudeSession>();
const geminiSessions = new Map<string, GeminiSession>();

function getOrCreateClaudeSession(threadId: string, opts?: { resumeSessionId?: string; cwd?: string }): ClaudeSession | null {
  if (!claudePath) return null;
  let session = claudeSessions.get(threadId);
  if (!session) {
    session = new ClaudeSession(claudePath, opts);
    claudeSessions.set(threadId, session);
  }
  return session;
}

function getOrCreateGeminiSession(threadId: string, opts?: { resumeSessionId?: string; cwd?: string }): GeminiSession | null {
  if (!geminiPath) return null;
  let session = geminiSessions.get(threadId);
  if (!session) {
    session = new GeminiSession(geminiPath, opts);
    geminiSessions.set(threadId, session);
  }
  return session;
}

function emitToRenderer(event: IpcEvent): void {
  mainWindow?.webContents.send("claude:event", event);
}

async function getProjectCwd(projectId: string): Promise<string | undefined> {
  const rows = await getDb()
    .select({ folderPath: projectFolders.folderPath })
    .from(projectFolders)
    .where(eq(projectFolders.projectId, projectId))
    .limit(1);
  return rows[0]?.folderPath;
}

function setupIpcHandlers(): void {
  ipcMain.handle("claude:check", () => ({ available: claudePath !== null, geminiAvailable: geminiPath !== null }));

  ipcMain.handle(
    "claude:send",
    async (_event, threadId: string, message: string, opts?: { provider?: Provider; resumeSessionId?: string; projectId?: string; images?: { name: string; base64: string; mediaType: string }[] }) => {
      let cwd: string | undefined;
      if (opts?.projectId) cwd = await getProjectCwd(opts.projectId);

      const provider: Provider = opts?.provider ?? "claude";
      emitToRenderer({ type: "status", threadId, status: "running" });

      if (provider === "gemini") {
        const session = getOrCreateGeminiSession(threadId, { resumeSessionId: opts?.resumeSessionId, cwd });
        if (!session) {
          emitToRenderer({ type: "error", threadId, message: "Gemini CLI not found on this machine." });
          return;
        }
        if (opts?.images?.length) {
          emitToRenderer({ type: "error", threadId, message: "This Gemini integration currently supports text prompts only." });
        }

        session.sendPrompt(message, (e) => {
          switch (e.kind) {
            case "init":
              emitToRenderer({ type: "session_init", threadId, sessionId: e.sessionId, model: e.model, provider: "gemini", cwd, projectId: opts?.projectId });
              break;
            case "text_delta":
              emitToRenderer({ type: "assistant_text", threadId, text: e.text });
              break;
            case "tool_use":
              emitToRenderer({ type: "tool_use", threadId, toolName: e.toolName, toolInput: e.toolInput });
              break;
            case "result":
              emitToRenderer({
                type: "result",
                threadId,
                costUsd: e.costUsd,
                durationMs: e.durationMs,
                ...(e.contextUsage?.totalTokens != null && { totalTokens: e.contextUsage.totalTokens }),
                ...(e.contextUsage?.maxTokens != null && { maxTokens: e.contextUsage.maxTokens }),
                ...(e.contextUsage?.percentage != null && { percentage: e.contextUsage.percentage }),
              });
              emitToRenderer({ type: "status", threadId, status: "idle" });
              break;
            case "error":
              emitToRenderer({ type: "error", threadId, message: e.message });
              break;
            case "done":
              emitToRenderer({ type: "status", threadId, status: "idle" });
              break;
          }
        });
        return;
      }

      const session = getOrCreateClaudeSession(threadId, { resumeSessionId: opts?.resumeSessionId, cwd });
      if (!session) {
        emitToRenderer({ type: "error", threadId, message: "Claude CLI not found on this machine." });
        return;
      }

      let currentSessionId: string | null = opts?.resumeSessionId ?? null;
      session.send(message, opts?.images, async (e) => {
        switch (e.kind) {
          case "init": {
            currentSessionId = e.sessionId;
            try {
              await getDb().insert(knownThreads).values({ sessionId: e.sessionId, createdAt: Date.now(), projectId: opts?.projectId ?? null }).onConflictDoNothing();
            } catch (err) {
              console.error("[db] failed to register session:", err);
              emitToRenderer({ type: "error", threadId, message: "Failed to save session to database." });
              return;
            }
            emitToRenderer({ type: "session_init", threadId, sessionId: e.sessionId, model: e.model, provider: "claude", cwd, projectId: opts?.projectId });
            break;
          }
          case "text_delta":
            emitToRenderer({ type: "assistant_text", threadId, text: e.text });
            break;
          case "tool_use":
            emitToRenderer({ type: "tool_use", threadId, toolName: e.toolName, toolInput: e.toolInput });
            break;
          case "result": {
            const contextStats = { ...(e.contextUsage != null && e.contextUsage), costUsd: e.costUsd, durationMs: e.durationMs };
            if (currentSessionId) {
              try {
                await getDb().update(knownThreads).set({ contextStats: JSON.stringify(contextStats) }).where(eq(knownThreads.sessionId, currentSessionId));
              } catch (err) {
                console.error("[db] failed to save context stats:", err);
              }
            }
            emitToRenderer({ type: "result", threadId, ...contextStats });
            if (e.deferredToolUse) {
              emitToRenderer({ type: "deferred_tool_use", threadId, toolUseId: e.deferredToolUse.id, toolName: e.deferredToolUse.name, toolInput: e.deferredToolUse.input });
              emitToRenderer({ type: "status", threadId, status: "awaiting_input" });
            } else {
              emitToRenderer({ type: "status", threadId, status: "idle" });
            }
            break;
          }
          case "error":
            emitToRenderer({ type: "error", threadId, message: e.message });
            break;
          case "done":
            emitToRenderer({ type: "status", threadId, status: "idle" });
            break;
        }
      });
    },
  );

  ipcMain.handle("claude:interrupt", (_event, threadId: string, provider?: Provider) => {
    const actualProvider = provider ?? "claude";
    if (actualProvider === "gemini") return geminiSessions.get(threadId)?.interrupt();
    return claudeSessions.get(threadId)?.interrupt();
  });

  ipcMain.handle("claude:respond-to-tool", (_event, threadId: string, toolUseId: string, responseText: string, provider?: Provider) => {
    if ((provider ?? "claude") !== "claude") return;
    const session = claudeSessions.get(threadId);
    if (!session) return;
    emitToRenderer({ type: "status", threadId, status: "running" });
    session.respondToTool(toolUseId, responseText);
  });

  ipcMain.handle("claude:list-sessions", async () => {
    const [all, known] = await Promise.all([
      listSessions(),
      getDb().select({ sessionId: knownThreads.sessionId, projectId: knownThreads.projectId, customName: knownThreads.customName, contextStats: knownThreads.contextStats }).from(knownThreads),
    ]);
    const knownMap = new Map(known.map((r) => [r.sessionId, { projectId: r.projectId ?? undefined, customName: r.customName ?? undefined, contextStats: r.contextStats ?? undefined }]));
    return all.filter((s) => knownMap.has(s.sessionId)).map((s) => {
      const row = knownMap.get(s.sessionId)!;
      let contextStats: object | undefined;
      try { contextStats = row.contextStats ? JSON.parse(row.contextStats) : undefined; } catch { /* ignore */ }
      return {
        sessionId: s.sessionId,
        summary: row.customName ?? s.summary,
        lastModified: s.lastModified,
        cwd: s.cwd,
        firstPrompt: s.firstPrompt,
        gitBranch: s.gitBranch,
        projectId: row.projectId,
        provider: "claude",
        contextStats,
      };
    });
  });

  ipcMain.handle("claude:load-history", async (_event, sessionId: string, provider?: Provider) => {
    if ((provider ?? "claude") !== "claude") return [];
    const msgs = await getSessionMessages(sessionId);

    const TextBlock = z.object({ type: z.literal("text"), text: z.string() });
    const ImageBlock = z.object({ type: z.literal("image"), source: z.object({ type: z.literal("base64"), media_type: z.string(), data: z.string() }) });
    const ToolUseBlock = z.object({ type: z.literal("tool_use"), id: z.string(), name: z.string(), input: z.record(z.string(), z.unknown()).optional() });
    const AnyBlock = z.union([TextBlock, ImageBlock, ToolUseBlock, z.object({ type: z.string() })]);
    const UserBody = z.object({ role: z.literal("user"), content: z.union([z.string(), z.array(AnyBlock)]) });
    const AssistantBody = z.object({ role: z.literal("assistant"), content: z.array(AnyBlock) });

    type HistoryImage = { mediaType: string; base64: string };
    type HistoryMsg =
      | { id: string; role: "user"; content: string; images?: HistoryImage[] }
      | { id: string; role: "assistant"; content: string }
      | { id: string; role: "tool_use"; toolName: string; toolInput?: Record<string, unknown> };

    const result: HistoryMsg[] = [];
    for (const m of msgs) {
      if (m.type === "user") {
        const parsed = UserBody.safeParse(m.message);
        if (!parsed.success) continue;
        let content: string;
        let images: HistoryImage[] | undefined;
        if (typeof parsed.data.content === "string") {
          content = parsed.data.content;
        } else {
          content = parsed.data.content.filter((b): b is z.infer<typeof TextBlock> => b.type === "text").map((b) => b.text).join("");
          const imageBlocks = parsed.data.content.filter((b): b is z.infer<typeof ImageBlock> => b.type === "image");
          if (imageBlocks.length > 0) images = imageBlocks.map((b) => ({ mediaType: b.source.media_type, base64: b.source.data }));
        }
        if (content.trim() || images) result.push({ id: m.uuid, role: "user", content, ...(images && { images }) });
      } else if (m.type === "assistant") {
        const parsed = AssistantBody.safeParse(m.message);
        if (!parsed.success) continue;
        const blocks = parsed.data.content;
        const textContent = blocks.filter((b): b is z.infer<typeof TextBlock> => b.type === "text").map((b) => b.text).join("");
        if (textContent.trim()) result.push({ id: m.uuid, role: "assistant", content: textContent });
        const toolBlocks = blocks.filter((b): b is z.infer<typeof ToolUseBlock> => b.type === "tool_use");
        for (const t of toolBlocks) result.push({ id: `${m.uuid}-${t.id}`, role: "tool_use", toolName: t.name, toolInput: t.input });
      }
    }

    return result;
  });

  ipcMain.handle("projects:create", async () => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, { properties: ["openDirectory"], message: "Choose a project folder" });
    if (result.canceled || result.filePaths.length === 0) return null;
    const folderPath = result.filePaths[0];
    const name = folderPath.split("/").filter(Boolean).at(-1) ?? folderPath;
    const projectId = randomUUID();
    const folderId = randomUUID();
    const now = Date.now();
    const db = getDb();
    await db.insert(projects).values({ id: projectId, name, createdAt: now });
    await db.insert(projectFolders).values({ id: folderId, projectId, folderPath, createdAt: now });
    return { id: projectId, name, folders: [{ id: folderId, path: folderPath }] };
  });

  ipcMain.handle("projects:list", async () => {
    const db = getDb();
    const [allProjects, allFolders] = await Promise.all([db.select().from(projects), db.select().from(projectFolders)]);
    return allProjects.map((p) => ({ id: p.id, name: p.name, folders: allFolders.filter((f) => f.projectId === p.id).map((f) => ({ id: f.id, path: f.folderPath })) }));
  });

  ipcMain.handle("projects:rename", async (_event, projectId: string, name: string) => {
    await getDb().update(projects).set({ name }).where(eq(projects.id, projectId));
  });

  ipcMain.handle("projects:delete", async (_event, projectId: string) => {
    const db = getDb();
    const threadRows = await db.select({ sessionId: knownThreads.sessionId }).from(knownThreads).where(eq(knownThreads.projectId, projectId));
    for (const row of threadRows) {
      const claudeSession = claudeSessions.get(row.sessionId);
      if (claudeSession) {
        claudeSession.close();
        claudeSessions.delete(row.sessionId);
      }
      const geminiSession = geminiSessions.get(row.sessionId);
      if (geminiSession) {
        geminiSession.close();
        geminiSessions.delete(row.sessionId);
      }
    }
    await db.delete(knownThreads).where(eq(knownThreads.projectId, projectId));
    await db.delete(projectFolders).where(eq(projectFolders.projectId, projectId));
    await db.delete(projects).where(eq(projects.id, projectId));
  });

  ipcMain.handle("projects:add-folder", async (_event, projectId: string, folderPath: string) => {
    const id = randomUUID();
    await getDb().insert(projectFolders).values({ id, projectId, folderPath, createdAt: Date.now() });
    return id;
  });

  ipcMain.handle("projects:remove-folder", async (_event, folderId: string) => {
    await getDb().delete(projectFolders).where(eq(projectFolders.id, folderId));
  });

  ipcMain.handle("git:branch", (_event, cwd: string): string | null => {
    try {
      return execSync("git rev-parse --abbrev-ref HEAD", { cwd, encoding: "utf-8" }).trim();
    } catch {
      return null;
    }
  });

  ipcMain.handle("threads:rename", async (_event, sessionId: string, name: string) => {
    await getDb().update(knownThreads).set({ customName: name }).where(eq(knownThreads.sessionId, sessionId));
  });

  ipcMain.handle("threads:delete", async (_event, sessionId: string) => {
    const claudeSession = claudeSessions.get(sessionId);
    if (claudeSession) {
      claudeSession.close();
      claudeSessions.delete(sessionId);
    }
    const geminiSession = geminiSessions.get(sessionId);
    if (geminiSession) {
      geminiSession.close();
      geminiSessions.delete(sessionId);
    }
    await getDb().delete(knownThreads).where(eq(knownThreads.sessionId, sessionId));
  });
}

function createWindow() {
  nativeTheme.themeSource = "dark";
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 780,
    minWidth: 800,
    minHeight: 500,
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 18 },
    transparent: true,
    vibrancy: "sidebar",
    visualEffectState: "active",
    roundedCorners: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true, preload: path.join(__dirname, "preload.js") },
    show: false,
  });

  mainWindow.once("ready-to-show", () => mainWindow!.show());
  mainWindow.on("closed", () => { mainWindow = null; });
  if (IS_DEV) mainWindow.loadURL(DEV_URL);
  else mainWindow.loadFile(path.join(__dirname, "../web/dist/index.html"));
}

function checkForUpdates(): void {
  autoUpdater.logger = console;
  autoUpdater.autoDownload = false;
  autoUpdater.on("update-available", (info) => {
    if (!mainWindow) return;
    dialog.showMessageBox(mainWindow, {
      type: "info",
      title: "Update Available",
      message: `Version ${info.version} is available.`,
      detail: "Would you like to open the download page?",
      buttons: ["Download", "Later"],
      defaultId: 0,
    }).then(({ response }) => {
      if (response === 0) shell.openExternal("https://github.com/alekseyrozh/openclawdex/releases/latest");
    });
  });
  autoUpdater.checkForUpdates();
}

app.whenReady().then(async () => {
  await initDb();
  setupIpcHandlers();
  createWindow();
  if (app.isPackaged) checkForUpdates();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  for (const session of claudeSessions.values()) session.close();
  for (const session of geminiSessions.values()) session.close();
  claudeSessions.clear();
  geminiSessions.clear();
});
