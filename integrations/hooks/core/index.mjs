// QuoteMem adapter core: zero-dependency Node (>= 20.3) modules shared by
// every host adapter (DSH, Claude Code, Codex, OpenCode, …).
//
// This folder is the source of truth. Each adapter ships a byte-identical
// copy (`node integrations/core/sync.mjs` writes them, `--check` verifies),
// because an installed plugin cannot reach files outside its own folder.
export { MAX_ID_CHARS, MIN_INJECT_CHARS, OFF, ON, flag, integer, osUser, present, publicSettings, resolveSettings } from './config.mjs';
export { REDACT_MARGIN, redactText } from './redact.mjs';
export { endsWithQuestion, isCommand, isMemoryRelated, isQuestion, isTrivial, sentences } from './gate.mjs';
export { FORGET_HEADER, MEMORY_OFF_HEADER, RECALL_HEADER, RESTARTING_HEADER, SETUP_HEADER, UNAVAILABLE_HEADER, setupHeader, chatForgetNote, fitContext, forgetNote, formatRecall, hitMemory, isEvidenceHit, plainForgetDone, unavailableNote } from './format.mjs';
export { authHeaders, callJson, checkedGet, codeOf, errKind, healthOk, postJson, retryAfterMs, retryable } from './http.mjs';
export { OWNER_LOOKUP_MS, checkListener, describeRefusal, listenerAddress, listenerConnection, listenerProof, onThisComputer, pinnedUrl, plainAddress, requestOn, thisComputerHost } from './listener.mjs';
export { buildJobs, cutBatch, newCaptureState, recordTurn, sha16, textOf, timeOf, wireMessage } from './capture.mjs';
export { ADD_HEADERS, AUTH_PAUSE_MS, backoffMs, chatForgetsOf, deferMs, pollDelayMs, rotateJob, sendAdd, validJob } from './send.mjs';
export { DIRECT_START_FILE, FOREIGN_STALE_MS, STARTED_DIRECTLY_VAR, alive, bootTime, homeDir, hostName, launcherCheck, ownerGone, ownerRecord, portOwner, portOwnerAsync, settleLaunches, spawnDetached, spawnDetachedAsync, userPort } from './proc.mjs';
export { adoptLock, handLock, lockHeld, readJson, releaseLock, stateKey, touchLock, tryLock, withLock, writeFileAtomic } from './lock.mjs';
export { fileSink, makeLog } from './log.mjs';
export { ACCEPTED_MAX_WAIT_MS, ACCEPTED_READ_TIMEOUT_MS, DiskQueue, FORGET_CONFLICT_ATTEMPTS, FORGET_EARLIER_WAIT_MS, FORGET_TOOL_WAIT_MS, FLUSHER_PATH, IDLE_LOCK_STALE_MS, PROBE_EVERY_MS, jobKey, ownerKey, runFlusher } from './diskqueue.mjs';
export { MAX_FORGET_STATEMENT_CHARS, carryOnDisk, detectForget, detectForgetRequests, forgetBody, forgetRequestId, memoryKey, requestKey, FORGOTTEN_KEEP_MS, forgottenFile, forgottenFiles, forgottenOf, noteForgotten, isForgetRequest, quotedMatch, runForget, saidIn, sameFact, statementsOf, withoutEnumerator } from './forget.mjs';
export { RecallBreaker, SEARCH_HITS, newRecallMemo, recallBlock, recallQuery, recallTurn, recentIds, rememberIds, searchBody, searchRecall } from './recall.mjs';
export { FORGET_WAIT_MS, HookMemory, IDLE_EXTRA_LIFE_MS, LISTENER_NOTE_MS, IDLE_PASS_MS, IDLE_TICK_MS, alignTurns, forgetWaitWithin, queueOwnerSessions, runIdleWaiter } from './hooks.mjs';
export { readTokenFile, resolveToken, serviceToken, serviceTokenFile } from './token.mjs';
export { CURSOR_CLI_CONTEXT_VERSION, CURSOR_CONTEXT_VERSION, cursorCliVersion, cursorPromptContext, foreignHost, markNative, nativeAdapterFor, nativeInstalled, nativeMarkers } from './native.mjs';
export { CONVERSATION_MAX_BYTES, conversationTurns, cursorStampTime, lastAssistantReply, lastAssistantText, promptHead, readConversation, replyFromLines, replyFromSteps, turnReply, turnsFromLines } from './transcript.mjs';
export {
  DEFAULT_PORT, REPLACE_AGAIN_MS, SERVICE_RELEASE, capLog, dataDirOf, ensureService, findUvx, isLoopbackUrl, isServiceCommand, localService,
  parseServiceEnv, readEnvFile, readServiceEnv, olderState, refused, replaceOlder, serveArgs, serviceEnvironment, servicePaths, serviceState, setupNote, stopService,
} from './autostart.mjs';
