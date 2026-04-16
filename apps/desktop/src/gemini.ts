import { execSync, spawn, type ChildProcessWithoutNullStreams } from "child_process";
import readline from "readline";

export function findGeminiBinary(): string | null {
  try {
    return execSync("which gemini", { encoding: "utf-8" }).trim();
  } catch {
    return null;
  }
}

export type ContextUsage = { totalTokens?: number; maxTokens?: number; percentage?: number };

export type SessionEvent =
  | { kind: "init"; sessionId: string; model: string }
  | { kind: "text_delta"; text: string }
  | { kind: "tool_use"; toolName: string; toolInput: Record<string, unknown> }
  | { kind: "result"; costUsd: number; durationMs: number; isError: boolean; contextUsage: ContextUsage | null }
  | { kind: "error"; message: string }
  | { kind: "done" };

type StreamEvent =
  | { type: "init"; session_id: string; model: string }
  | { type: "message"; role: "user" | "assistant"; content: string; delta?: boolean }
  | { type: "tool_use"; tool_name: string; tool_id: string; parameters: Record<string, unknown> }
  | { type: "tool_result"; tool_id: string; status: "success" | "error"; output?: string; error?: { type: string; message: string } }
  | { type: "error"; severity: "warning" | "error"; message: string }
  | { type: "result"; status: "success" | "error"; error?: { type: string; message: string }; stats?: { total_tokens?: number; input_tokens?: number; output_tokens?: number; duration_ms?: number } };

export class GeminiSession {
  private geminiPath: string;
  private resumeSessionId: string | undefined;
  private cwd: string | undefined;
  private child: ChildProcessWithoutNullStreams | null = null;

  constructor(geminiPath: string, opts?: { resumeSessionId?: string; cwd?: string }) {
    this.geminiPath = geminiPath;
    this.resumeSessionId = opts?.resumeSessionId;
    this.cwd = opts?.cwd;
  }

  send(_message: string, _images: unknown, onEvent: (e: SessionEvent) => void): void {
    throw new Error("Use sendPrompt() for GeminiSession");
  }

  sendPrompt(message: string, onEvent: (e: SessionEvent) => void): void {
    const args = ["-p", message, "--output-format", "stream-json", "--approval-mode", "yolo"];

    if (this.resumeSessionId) {
      args.unshift(this.resumeSessionId);
      args.unshift("--resume");
    }

    this.child = spawn(this.geminiPath, args, {
      cwd: this.cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    const rl = readline.createInterface({ input: this.child.stdout });

    rl.on("line", (line) => {
      if (!line.trim()) return;

      let evt: StreamEvent;
      try {
        evt = JSON.parse(line) as StreamEvent;
      } catch {
        return;
      }

      switch (evt.type) {
        case "init": {
          this.resumeSessionId = evt.session_id;
          onEvent({ kind: "init", sessionId: evt.session_id, model: evt.model });
          break;
        }
        case "message": {
          if (evt.role === "assistant" && evt.content) {
            onEvent({ kind: "text_delta", text: evt.content });
          }
          break;
        }
        case "tool_use": {
          onEvent({ kind: "tool_use", toolName: evt.tool_name, toolInput: evt.parameters ?? {} });
          break;
        }
        case "error": {
          onEvent({ kind: "error", message: evt.message });
          break;
        }
        case "result": {
          const totalTokens = (evt.stats?.input_tokens ?? 0) + (evt.stats?.output_tokens ?? 0);
          onEvent({
            kind: "result",
            costUsd: 0,
            durationMs: evt.stats?.duration_ms ?? 0,
            isError: evt.status === "error",
            contextUsage: totalTokens > 0 ? { totalTokens } : null,
          });
          if (evt.status === "error" && evt.error?.message) {
            onEvent({ kind: "error", message: evt.error.message });
          }
          break;
        }
      }
    });

    this.child.stderr.on("data", (buf) => {
      const text = String(buf).trim();
      if (text) onEvent({ kind: "error", message: text });
    });

    this.child.on("close", () => {
      onEvent({ kind: "done" });
      this.child = null;
    });
  }

  async interrupt(): Promise<void> {
    try {
      this.child?.kill("SIGINT");
    } catch {
      /* ignore */
    }
  }

  close(): void {
    try {
      this.child?.kill("SIGTERM");
    } catch {
      /* ignore */
    }
    this.child = null;
  }
}
