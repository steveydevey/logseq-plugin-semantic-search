import { normalizeContent, hashContent, truncate } from "./utils";
import { embedTexts } from "./embeddings";
import {
  getEmbedding,
  putEmbeddings,
  getAllEmbeddings,
  deleteEmbeddings,
  clearAllEmbeddings,
  getMetadata,
  setMetadata,
  getEmbeddingCount,
} from "./storage";
import { getSettings } from "./settings";

export interface IndexingState {
  status: "idle" | "indexing";
  progress: { done: number; total: number };
}

export const indexingState: IndexingState = {
  status: "idle",
  progress: { done: 0, total: 0 },
};

let currentAbort: AbortController | null = null;

interface BlockResult {
  id: number;
  uuid: string;
  content: string;
  page?: { id: number };
}

export function cancelIndexing(): void {
  if (currentAbort) {
    currentAbort.abort();
    currentAbort = null;
  }
}

export async function indexBlocks(
  onProgress?: (done: number, total: number) => void,
): Promise<void> {
  cancelIndexing();

  const abort = new AbortController();
  currentAbort = abort;

  const settings = getSettings();

  try {
    // Check if model changed
    const storedModel = await getMetadata("model");
    if (storedModel && storedModel !== settings.embeddingModel) {
      await clearAllEmbeddings();
      logseq.UI.showMsg("Model changed, re-indexing all blocks...");
    }
    await setMetadata("model", settings.embeddingModel);

    // Query all blocks
    const query = `[:find (pull ?b [:db/id :block/uuid :block/content :block/page])
     :where [?b :block/content ?c] [(>= (count ?c) ${settings.minBlockLength})]]`;
    const results: BlockResult[][] = await logseq.DB.datascriptQuery(query);
    const blocks = results.map((r) => r[0]);

    indexingState.status = "indexing";
    indexingState.progress = { done: 0, total: blocks.length };

    // Find blocks needing embedding
    const toEmbed: {
      blockId: string;
      content: string;
      hash: string;
      pageId: number;
    }[] = [];

    const currentBlockIds = new Set<string>();

    for (const block of blocks) {
      if (abort.signal.aborted) return;

      const blockId = block.uuid;
      if (!blockId || !block.content) continue;
      currentBlockIds.add(blockId);

      const normalized = normalizeContent(block.content);
      if (normalized.length < settings.minBlockLength) continue;

      const hash = await hashContent(normalized);
      const existing = await getEmbedding(blockId);

      if (!existing || existing.contentHash !== hash) {
        toEmbed.push({
          blockId,
          content: truncate(normalized, 4000),
          hash,
          pageId: block.page?.id ?? 0,
        });
      }
    }

    // Batch embed
    const total = toEmbed.length;
    let done = 0;

    for (let i = 0; i < toEmbed.length; i += settings.batchSize) {
      if (abort.signal.aborted) return;

      const batch = toEmbed.slice(i, i + settings.batchSize);
      const texts = batch.map((b) => b.content);

      const embeddings = await embedTexts(
        texts,
        settings.apiEndpoint,
        settings.embeddingModel,
        settings.apiFormat,
        abort.signal,
      );

      const records = batch.map((b, j) => ({
        blockId: b.blockId,
        contentHash: b.hash,
        embedding: embeddings[j],
        pageId: b.pageId,
        timestamp: Date.now(),
      }));

      await putEmbeddings(records);

      done += batch.length;
      indexingState.progress = { done, total };
      onProgress?.(done, total);

      // Yield to event loop
      await new Promise((r) => setTimeout(r, 0));
    }

    // Delete stale entries
    const allStored = await getAllEmbeddings();
    const staleIds = allStored
      .map((r) => r.blockId)
      .filter((id) => !currentBlockIds.has(id));
    if (staleIds.length > 0) {
      await deleteEmbeddings(staleIds);
    }

    const count = await getEmbeddingCount();
    await setMetadata("blockCount", count);
    await setMetadata("lastIndexed", Date.now());

    if (!abort.signal.aborted && total > 0) {
      logseq.UI.showMsg(`Indexed ${total} blocks`);
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    const msg = (err as Error).message || "Indexing failed";
    logseq.UI.showMsg(msg, "error");
    throw err;
  } finally {
    indexingState.status = "idle";
    if (currentAbort === abort) {
      currentAbort = null;
    }
  }
}
