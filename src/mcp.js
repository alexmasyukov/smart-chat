import { readFile } from "node:fs/promises";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// Делает имя пригодным для function-calling OpenAI: ^[a-zA-Z0-9_-]+$
function sanitize(name) {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_");
}

export class McpHub {
  constructor() {
    this.clients = []; // { name, client }
    // mapping: имя функции в OpenAI -> { client, toolName }
    this.toolMap = new Map();
    this.openaiTools = []; // массив в формате tools для OpenAI
  }

  // Поднимает все серверы из конфига и собирает их инструменты.
  async loadFromConfig(configPath) {
    let raw;
    try {
      raw = await readFile(configPath, "utf8");
    } catch {
      console.warn(`Конфиг MCP не найден (${configPath}) — работаем без инструментов.`);
      return;
    }

    const config = JSON.parse(raw);
    const servers = config.mcpServers || {};

    for (const [serverName, conf] of Object.entries(servers)) {
      try {
        await this.connectServer(serverName, conf);
      } catch (err) {
        console.warn(`Не удалось подключить MCP-сервер "${serverName}": ${err.message}`);
      }
    }
  }

  async connectServer(serverName, conf) {
    const transport = new StdioClientTransport({
      command: conf.command,
      args: conf.args || [],
      env: { ...process.env, ...(conf.env || {}) },
    });

    const client = new Client(
      { name: "smart-chat", version: "1.0.0" },
      { capabilities: {} }
    );

    await client.connect(transport);
    this.clients.push({ name: serverName, client });

    const { tools } = await client.listTools();
    for (const tool of tools) {
      const fnName = sanitize(`${serverName}__${tool.name}`).slice(0, 64);
      this.toolMap.set(fnName, { client, toolName: tool.name });
      this.openaiTools.push({
        type: "function",
        function: {
          name: fnName,
          description: tool.description || `Инструмент ${tool.name} из ${serverName}`,
          parameters: tool.inputSchema || { type: "object", properties: {} },
        },
      });
    }

    // Лог в stderr: stdout зарезервирован под NDJSON-протокол sidecar-движка.
    console.error(`  ✓ MCP "${serverName}": ${tools.length} инструм.`);
  }

  hasTools() {
    return this.openaiTools.length > 0;
  }

  // Выполняет вызов инструмента по имени функции от модели.
  async callTool(fnName, args) {
    const entry = this.toolMap.get(fnName);
    if (!entry) return `Ошибка: инструмент "${fnName}" не найден.`;

    const result = await entry.client.callTool({
      name: entry.toolName,
      arguments: args || {},
    });

    // Сводим content (text/image/...) к тексту для модели.
    const parts = (result.content || []).map((c) => {
      if (c.type === "text") return c.text;
      return `[${c.type}]`;
    });
    const text = parts.join("\n") || "(пустой результат)";
    return result.isError ? `Ошибка инструмента: ${text}` : text;
  }

  async close() {
    for (const { client } of this.clients) {
      try {
        await client.close();
      } catch {
        // игнорируем ошибки закрытия
      }
    }
  }
}
