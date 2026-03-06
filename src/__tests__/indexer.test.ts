import "fake-indexeddb/auto";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock @logseq/libs
const mockLogseq = {
  settings: {
    apiEndpoint: "http://localhost:11434",
    apiFormat: "ollama",
    embeddingModel: "nomic-embed-text",
    batchSize: 2,
    topK: 20,
    minBlockLength: 10,
    autoIndexOnLoad: true,
  },
  DB: {
    datascriptQuery: vi.fn(),
  },
  UI: {
    showMsg: vi.fn(),
  },
};
vi.stubGlobal("logseq", mockLogseq);

// Mock embedTexts
vi.mock("../embeddings", () => ({
  embedTexts: vi.fn(),
}));

import { indexBlocks, indexingState, cancelIndexing } from "../indexer";
import { embedTexts } from "../embeddings";
import { getAllEmbeddings, setMetadata } from "../storage";

const mockEmbedTexts = vi.mocked(embedTexts);

beforeEach(async () => {
  vi.clearAllMocks();
  indexingState.status = "idle";
  indexingState.progress = { done: 0, total: 0 };

  // Clear IndexedDB
  const dbs = await indexedDB.databases();
  for (const db of dbs) {
    if (db.name) indexedDB.deleteDatabase(db.name);
  }
});

describe("indexBlocks", () => {
  it("skips unchanged blocks", async () => {
    mockLogseq.DB.datascriptQuery.mockResolvedValue([
      [{ id: 1, uuid: "u1", content: "Hello world this is a test block", page: { id: 10 } }],
    ]);

    mockEmbedTexts.mockResolvedValue([[0.1, 0.2, 0.3]]);

    // First index
    await indexBlocks();
    expect(mockEmbedTexts).toHaveBeenCalledTimes(1);

    // Second index - same content, should skip
    mockEmbedTexts.mockClear();
    await indexBlocks();
    expect(mockEmbedTexts).not.toHaveBeenCalled();
  });

  it("re-embeds changed blocks", async () => {
    mockLogseq.DB.datascriptQuery.mockResolvedValueOnce([
      [{ id: 1, uuid: "u1", content: "Hello world this is a test block", page: { id: 10 } }],
    ]);
    mockEmbedTexts.mockResolvedValue([[0.1, 0.2, 0.3]]);

    await indexBlocks();

    // Change content
    mockLogseq.DB.datascriptQuery.mockResolvedValueOnce([
      [{ id: 1, uuid: "u1", content: "Changed content that is different", page: { id: 10 } }],
    ]);
    mockEmbedTexts.mockClear();
    mockEmbedTexts.mockResolvedValue([[0.4, 0.5, 0.6]]);

    await indexBlocks();
    expect(mockEmbedTexts).toHaveBeenCalledTimes(1);
  });

  it("batches correctly", async () => {
    // batchSize is 2
    mockLogseq.DB.datascriptQuery.mockResolvedValue([
      [{ id: 1, uuid: "u1", content: "First block with enough content", page: { id: 10 } }],
      [{ id: 2, uuid: "u2", content: "Second block with enough content", page: { id: 10 } }],
      [{ id: 3, uuid: "u3", content: "Third block with enough content", page: { id: 11 } }],
    ]);

    mockEmbedTexts
      .mockResolvedValueOnce([[0.1], [0.2]])  // batch 1
      .mockResolvedValueOnce([[0.3]]);          // batch 2

    await indexBlocks();

    expect(mockEmbedTexts).toHaveBeenCalledTimes(2);
    const stored = await getAllEmbeddings();
    expect(stored).toHaveLength(3);
  });

  it("tracks progress", async () => {
    mockLogseq.DB.datascriptQuery.mockResolvedValue([
      [{ id: 1, uuid: "u1", content: "Block one with enough content", page: { id: 10 } }],
      [{ id: 2, uuid: "u2", content: "Block two with enough content", page: { id: 10 } }],
    ]);

    mockEmbedTexts.mockResolvedValue([[0.1], [0.2]]);

    const progress: Array<{ done: number; total: number }> = [];
    await indexBlocks((done, total) => progress.push({ done, total }));

    expect(progress.length).toBeGreaterThan(0);
    expect(progress[progress.length - 1].done).toBe(2);
  });

  it("clears embeddings on model change", async () => {
    mockLogseq.DB.datascriptQuery.mockResolvedValue([
      [{ id: 1, uuid: "u1", content: "Hello world this is a test block", page: { id: 10 } }],
    ]);
    mockEmbedTexts.mockResolvedValue([[0.1, 0.2]]);

    await indexBlocks();
    const count1 = (await getAllEmbeddings()).length;
    expect(count1).toBe(1);

    // Simulate model change via stored metadata
    await setMetadata("model", "different-model");

    mockEmbedTexts.mockClear();
    mockEmbedTexts.mockResolvedValue([[0.3, 0.4]]);

    await indexBlocks();
    expect(mockEmbedTexts).toHaveBeenCalled();
  });

  it("handles cancellation", async () => {
    mockLogseq.DB.datascriptQuery.mockResolvedValue([
      [{ id: 1, uuid: "u1", content: "Block one with enough content", page: { id: 10 } }],
    ]);

    mockEmbedTexts.mockImplementation(async () => {
      cancelIndexing();
      throw new DOMException("Aborted", "AbortError");
    });

    // Should not throw
    await indexBlocks();
    expect(indexingState.status).toBe("idle");
  });
});
